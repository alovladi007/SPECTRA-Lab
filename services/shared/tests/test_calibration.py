"""
Unit tests for calibration validation logic.
"""

import pytest
from datetime import datetime, timedelta, timezone

from services.shared.db.models import (
    Calibration, CalibrationStatus, Instrument, InstrumentStatus
)
from services.shared.db.deps import check_instrument_calibration


class TestCalibrationValidation:
    """Test calibration validation logic."""

    def test_valid_calibration(self, db_session, instrument_with_valid_cal):
        """Test instrument with valid calibration returns valid status."""
        result = check_instrument_calibration(str(instrument_with_valid_cal.id), db_session)

        assert result["valid"] is True
        assert result["status"] == "valid"
        assert "latest_calibration" in result
        assert "expires_at" in result["latest_calibration"]
        assert "certificate_id" in result["latest_calibration"]

    def test_expired_calibration(self, db_session, instrument_with_expired_cal):
        """Test instrument with expired calibration returns invalid status."""
        result = check_instrument_calibration(str(instrument_with_expired_cal.id), db_session)

        assert result["valid"] is False
        assert result["status"] == "expired"
        assert "latest_calibration" in result
        assert result["latest_calibration"]["certificate_id"] == "CAL-2024-999"

    def test_no_calibration(self, db_session, instrument_no_cal):
        """Test instrument without calibration returns no_calibration status."""
        result = check_instrument_calibration(str(instrument_no_cal.id), db_session)

        assert result["valid"] is False
        assert result["status"] == "no_calibration"
        assert "latest_calibration" not in result

    def test_multiple_calibrations_uses_latest(self, db_session, org1):
        """Test that latest calibration is used when multiple exist."""
        # Create instrument
        instrument = Instrument(
            organization_id=org1.id,
            name="Multi-Cal Instrument",
            vendor="Thermo",
            model="Nicolet iS50",
            serial="FTIR-001",
            interface="visa_usb",
            status=InstrumentStatus.ONLINE
        )
        db_session.add(instrument)
        db_session.commit()
        db_session.refresh(instrument)

        now = datetime.now(timezone.utc)

        # Add old expired calibration
        old_cal = Calibration(
            instrument_id=instrument.id,
            certificate_id="CAL-OLD",
            issued_at=now - timedelta(days=400),
            expires_at=now - timedelta(days=35),
            status=CalibrationStatus.EXPIRED
        )
        db_session.add(old_cal)

        # Add recent valid calibration
        new_cal = Calibration(
            instrument_id=instrument.id,
            certificate_id="CAL-NEW",
            issued_at=now - timedelta(days=10),
            expires_at=now + timedelta(days=355),
            status=CalibrationStatus.VALID
        )
        db_session.add(new_cal)
        db_session.commit()

        # Should use the latest (most recent issued_at)
        result = check_instrument_calibration(str(instrument.id), db_session)

        assert result["valid"] is True
        assert result["status"] == "valid"
        assert result["latest_calibration"]["certificate_id"] == "CAL-NEW"

    def test_calibration_about_to_expire(self, db_session, org1):
        """Test calibration expiring soon is still valid."""
        instrument = Instrument(
            organization_id=org1.id,
            name="Soon-Expire Instrument",
            vendor="Generic", model="Test Model", interface="visa_usb",
            serial="ELLIP-001",
            status=InstrumentStatus.ONLINE
        )
        db_session.add(instrument)
        db_session.commit()
        db_session.refresh(instrument)

        now = datetime.now(timezone.utc)

        # Calibration expiring in 1 day
        calibration = Calibration(
            instrument_id=instrument.id,
            certificate_id="CAL-EXPIRING",
            issued_at=now - timedelta(days=364),
            expires_at=now + timedelta(days=1),
            status=CalibrationStatus.VALID
        )
        db_session.add(calibration)
        db_session.commit()

        # Should still be valid until it actually expires
        result = check_instrument_calibration(str(instrument.id), db_session)

        assert result["valid"] is True
        assert result["status"] == "valid"

    def test_calibration_expired_by_one_second(self, db_session, org1):
        """Test calibration that just expired is invalid."""
        instrument = Instrument(
            organization_id=org1.id,
            name="Just-Expired Instrument",
            vendor="Generic", model="Test Model", interface="visa_usb",
            serial="PROF-001",
            status=InstrumentStatus.ONLINE
        )
        db_session.add(instrument)
        db_session.commit()
        db_session.refresh(instrument)

        now = datetime.now(timezone.utc)

        # Calibration expired 1 second ago
        calibration = Calibration(
            instrument_id=instrument.id,
            certificate_id="CAL-JUST-EXPIRED",
            issued_at=now - timedelta(days=365),
            expires_at=now - timedelta(seconds=1),
            status=CalibrationStatus.EXPIRED
        )
        db_session.add(calibration)
        db_session.commit()

        result = check_instrument_calibration(str(instrument.id), db_session)

        assert result["valid"] is False
        assert result["status"] == "expired"

    def test_revoked_calibration_invalid(self, db_session, org1):
        """Test revoked calibration is treated as invalid."""
        instrument = Instrument(
            organization_id=org1.id,
            name="Revoked Cal Instrument",
            vendor="Generic", model="Test Model", interface="visa_usb",
            serial="AFM-001",
            status=InstrumentStatus.ONLINE
        )
        db_session.add(instrument)
        db_session.commit()
        db_session.refresh(instrument)

        now = datetime.now(timezone.utc)

        # Valid calibration but revoked
        calibration = Calibration(
            instrument_id=instrument.id,
            certificate_id="CAL-REVOKED",
            issued_at=now - timedelta(days=30),
            expires_at=now + timedelta(days=335),
            status=CalibrationStatus.REVOKED
        )
        db_session.add(calibration)
        db_session.commit()

        result = check_instrument_calibration(str(instrument.id), db_session)

        # Even though expires_at is in future, status is REVOKED
        assert result["valid"] is False

    def test_pending_calibration_invalid(self, db_session, org1):
        """Test pending calibration is not yet valid."""
        instrument = Instrument(
            organization_id=org1.id,
            name="Pending Cal Instrument",
            vendor="Generic", model="Test Model", interface="visa_usb",
            serial="SIMS-002",
            status=InstrumentStatus.ONLINE
        )
        db_session.add(instrument)
        db_session.commit()
        db_session.refresh(instrument)

        now = datetime.now(timezone.utc)

        # Calibration pending verification
        calibration = Calibration(
            instrument_id=instrument.id,
            certificate_id="CAL-PENDING",
            issued_at=now,
            expires_at=now + timedelta(days=365),
            status=CalibrationStatus.PENDING
        )
        db_session.add(calibration)
        db_session.commit()

        result = check_instrument_calibration(str(instrument.id), db_session)

        # Pending calibration should not be considered valid
        assert result["valid"] is False


class TestCalibrationWorkflow:
    """Test calibration lifecycle workflows."""

    def test_new_calibration_replaces_expired(self, db_session, instrument_with_expired_cal):
        """Test adding new calibration to instrument with expired cal."""
        # Initially expired
        result = check_instrument_calibration(str(instrument_with_expired_cal.id), db_session)
        assert result["valid"] is False
        assert result["status"] == "expired"

        # Add new valid calibration
        now = datetime.now(timezone.utc)
        new_cal = Calibration(
            instrument_id=instrument_with_expired_cal.id,
            certificate_id="CAL-2025-NEW",
            issued_at=now,
            expires_at=now + timedelta(days=365),
            status=CalibrationStatus.VALID
        )
        db_session.add(new_cal)
        db_session.commit()

        # Should now be valid
        result = check_instrument_calibration(str(instrument_with_expired_cal.id), db_session)
        assert result["valid"] is True
        assert result["status"] == "valid"
        assert result["latest_calibration"]["certificate_id"] == "CAL-2025-NEW"

    def test_expiring_calibration_status_update(self, db_session, org1):
        """Test updating calibration status when it expires."""
        instrument = Instrument(
            organization_id=org1.id,
            name="Status Update Instrument",
            vendor="Generic", model="Test Model", interface="visa_usb",
            serial="XRD-002",
            status=InstrumentStatus.ONLINE
        )
        db_session.add(instrument)
        db_session.commit()
        db_session.refresh(instrument)

        now = datetime.now(timezone.utc)

        # Create calibration that already expired
        calibration = Calibration(
            instrument_id=instrument.id,
            certificate_id="CAL-TO-EXPIRE",
            issued_at=now - timedelta(days=400),
            expires_at=now - timedelta(days=10),
            status=CalibrationStatus.VALID  # Still marked valid in DB
        )
        db_session.add(calibration)
        db_session.commit()

        # Check should detect expiry based on expires_at, not status
        result = check_instrument_calibration(str(instrument.id), db_session)

        # Should be invalid because expires_at is past, even though status is VALID
        assert result["valid"] is False

    def test_calibration_metadata(self, db_session, instrument_with_valid_cal):
        """Test that calibration metadata is accessible."""
        result = check_instrument_calibration(str(instrument_with_valid_cal.id), db_session)

        assert result["valid"] is True

        # Get actual calibration from DB to verify metadata
        cal = db_session.query(Calibration).filter(
            Calibration.instrument_id == instrument_with_valid_cal.id
        ).first()

        assert cal.extra_metadata == {"accuracy": "±0.1%"}
        assert cal.performed_by == "Metrology Lab Inc."
