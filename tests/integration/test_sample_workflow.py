"""
Integration test: Complete sample creation workflow with auth and org scoping.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone

from services.lims.app.main import app as lims_app
from services.shared.auth.jwt import create_token_pair


pytestmark = pytest.mark.integration


class TestSampleCreationWorkflow:
    """Test complete workflow: login → create sample → verify org scoping."""

    def test_engineer_can_create_sample(self, engineer_user, engineer_token):
        """Engineer creates a sample successfully."""
        client = TestClient(lims_app)

        # Create sample
        response = client.post(
            "/api/samples",
            json={
                "name": "Integration Test Sample",
                "barcode": "INT-TEST-001",
                "sample_type": "wafer",
                "location": "Clean Room A"
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Integration Test Sample"
        assert data["barcode"] == "INT-TEST-001"
        assert data["organization_id"] == str(engineer_user.organization_id)

        sample_id = data["id"]

        # Verify sample appears in list
        response = client.get(
            "/api/samples",
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 200
        samples = response.json()
        assert any(s["id"] == sample_id for s in samples)

    def test_viewer_cannot_create_sample(self, viewer_user, viewer_token):
        """Viewer (read-only) cannot create samples."""
        client = TestClient(lims_app)

        response = client.post(
            "/api/samples",
            json={
                "name": "Should Fail",
                "barcode": "FAIL-001",
                "sample_type": "wafer"
            },
            headers={"Authorization": f"Bearer {viewer_token['access_token']}"}
        )

        assert response.status_code == 403  # Forbidden

    def test_cross_org_sample_isolation(self, engineer_user, org2_engineer, engineer_token):
        """Users from different orgs cannot see each other's samples."""
        client = TestClient(lims_app)

        # Org1 engineer creates sample
        response = client.post(
            "/api/samples",
            json={
                "name": "Org1 Secret Sample",
                "barcode": "ORG1-SECRET",
                "sample_type": "wafer"
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 201
        org1_sample_id = response.json()["id"]

        # Org2 engineer queries samples
        org2_token = create_token_pair(
            user_id=str(org2_engineer.id),
            org_id=str(org2_engineer.organization_id),
            role=org2_engineer.role.value,
            email=org2_engineer.email
        )

        response = client.get(
            "/api/samples",
            headers={"Authorization": f"Bearer {org2_token['access_token']}"}
        )

        assert response.status_code == 200
        org2_samples = response.json()

        # Org2 engineer should NOT see Org1's sample
        assert not any(s["id"] == org1_sample_id for s in org2_samples)

    def test_soft_delete_workflow(self, engineer_user, engineer_token, db_session):
        """Engineer can soft delete a sample and it no longer appears in default list."""
        client = TestClient(lims_app)

        # Create sample
        response = client.post(
            "/api/samples",
            json={
                "name": "Sample to Delete",
                "barcode": "DELETE-ME",
                "sample_type": "wafer"
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        sample_id = response.json()["id"]

        # Delete sample
        response = client.delete(
            f"/api/samples/{sample_id}",
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 204

        # Sample should not appear in default list
        response = client.get(
            "/api/samples",
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        samples = response.json()
        assert not any(s["id"] == sample_id for s in samples)

        # Sample should appear when including deleted
        response = client.get(
            "/api/samples?include_deleted=true",
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        samples = response.json()
        deleted_sample = next((s for s in samples if s["id"] == sample_id), None)
        assert deleted_sample is not None
        assert deleted_sample["is_deleted"] is True
