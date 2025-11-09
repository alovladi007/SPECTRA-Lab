"""
Integration test: Calibration lockout workflow with automatic run blocking.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone

from services.analysis.app.main import app as analysis_app
from services.shared.db.models import RunStatus


pytestmark = pytest.mark.integration


class TestCalibrationLockoutWorkflow:
    """Test calibration validation and automatic run blocking."""

    def test_run_blocked_on_expired_calibration(self, engineer_token, instrument_with_expired_cal, db_session):
        """Creating a run with expired calibration automatically blocks it."""
        client = TestClient(analysis_app)

        # Attempt to create run with instrument that has expired calibration
        response = client.post(
            "/api/v1/runs",
            json={
                "instrument_id": str(instrument_with_expired_cal.id),
                "recipe_name": "Standard SIMS Profile",
                "parameters": {"voltage": 5000, "current": 100}
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 201
        data = response.json()

        # Run should be created but in BLOCKED status
        assert data["status"] == "BLOCKED"
        assert data["blocked_reason"] is not None
        assert "calibration expired" in data["blocked_reason"].lower()
        assert data["instrument_id"] == str(instrument_with_expired_cal.id)

    def test_run_queued_on_valid_calibration(self, engineer_token, instrument_with_valid_cal):
        """Creating a run with valid calibration allows it to proceed."""
        client = TestClient(analysis_app)

        response = client.post(
            "/api/v1/runs",
            json={
                "instrument_id": str(instrument_with_valid_cal.id),
                "recipe_name": "Standard SIMS Profile",
                "parameters": {"voltage": 5000, "current": 100}
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 201
        data = response.json()

        # Run should be created in QUEUED status (ready to execute)
        assert data["status"] == "QUEUED"
        assert data["blocked_reason"] is None
        assert data["instrument_id"] == str(instrument_with_valid_cal.id)

    def test_run_blocked_on_missing_calibration(self, engineer_token, instrument_no_cal):
        """Creating a run with no calibration on record blocks it."""
        client = TestClient(analysis_app)

        response = client.post(
            "/api/v1/runs",
            json={
                "instrument_id": str(instrument_no_cal.id),
                "recipe_name": "Standard SEM Imaging",
                "parameters": {"magnification": 10000}
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 201
        data = response.json()

        # Run should be blocked due to no calibration
        assert data["status"] == "BLOCKED"
        assert "no calibration certificate" in data["blocked_reason"].lower()

    def test_unblock_run_after_calibration_upload(
        self, engineer_token, instrument_with_expired_cal, db_session
    ):
        """Blocked run can be unblocked after uploading new calibration."""
        client = TestClient(analysis_app)

        # Create blocked run
        response = client.post(
            "/api/v1/runs",
            json={
                "instrument_id": str(instrument_with_expired_cal.id),
                "recipe_name": "Test Run",
                "parameters": {}
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        run_id = response.json()["id"]
        assert response.json()["status"] == "BLOCKED"

        # Upload new valid calibration certificate
        now = datetime.now(timezone.utc)
        response = client.post(
            "/api/v1/calibrations",
            json={
                "instrument_id": str(instrument_with_expired_cal.id),
                "certificate_id": "CAL-2025-NEW",
                "issued_at": now.isoformat(),
                "expires_at": (now + timedelta(days=365)).isoformat(),
                "performed_by": "Metrology Lab Inc."
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 201

        # Attempt to unblock run
        response = client.post(
            f"/api/v1/runs/{run_id}/unblock",
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 200
        data = response.json()

        # Run should now be unblocked
        assert data["status"] == "QUEUED"
        assert data["blocked_reason"] is None

    def test_calibration_check_endpoint(self, engineer_token, instrument_with_valid_cal, instrument_with_expired_cal):
        """Calibration status check endpoint returns correct status."""
        client = TestClient(analysis_app)

        # Check valid calibration
        response = client.get(
            f"/api/v1/calibrations/status/check?instrument_id={instrument_with_valid_cal.id}",
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["status"] == "valid"
        assert "latest_calibration" in data

        # Check expired calibration
        response = client.get(
            f"/api/v1/calibrations/status/check?instrument_id={instrument_with_expired_cal.id}",
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["status"] == "expired"

    def test_list_blocked_runs(self, engineer_token, instrument_with_expired_cal, instrument_with_valid_cal):
        """Can filter runs by blocked status."""
        client = TestClient(analysis_app)

        # Create one blocked run
        client.post(
            "/api/v1/runs",
            json={
                "instrument_id": str(instrument_with_expired_cal.id),
                "recipe_name": "Blocked Run",
                "parameters": {}
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        # Create one queued run
        client.post(
            "/api/v1/runs",
            json={
                "instrument_id": str(instrument_with_valid_cal.id),
                "recipe_name": "Queued Run",
                "parameters": {}
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        # Filter by BLOCKED status
        response = client.get(
            "/api/v1/runs?status=BLOCKED",
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 200
        blocked_runs = response.json()
        assert all(run["status"] == "BLOCKED" for run in blocked_runs)
        assert any("Blocked Run" in run["recipe_name"] for run in blocked_runs)

    def test_admin_can_manually_expire_calibration(
        self, admin_token, instrument_with_valid_cal, engineer_token, db_session
    ):
        """Admin can manually expire a calibration, blocking future runs."""
        client = TestClient(analysis_app)

        # Get current calibration
        response = client.get(
            "/api/v1/calibrations",
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        calibrations = response.json()
        cal = next((c for c in calibrations if c["instrument_id"] == str(instrument_with_valid_cal.id)), None)
        assert cal is not None
        cal_id = cal["id"]

        # Admin manually expires calibration
        response = client.patch(
            f"/api/v1/calibrations/{cal_id}/expire",
            headers={"Authorization": f"Bearer {admin_token['access_token']}"}
        )

        assert response.status_code == 200

        # New run should now be blocked
        response = client.post(
            "/api/v1/runs",
            json={
                "instrument_id": str(instrument_with_valid_cal.id),
                "recipe_name": "Should Be Blocked",
                "parameters": {}
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 201
        assert response.json()["status"] == "BLOCKED"
