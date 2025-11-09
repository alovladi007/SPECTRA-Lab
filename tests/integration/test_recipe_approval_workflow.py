"""
Integration test: Recipe approval workflow with role-based access control.
"""

import pytest
from fastapi.testclient import TestClient

from services.lims.app.main import app as lims_app


pytestmark = pytest.mark.integration


class TestRecipeApprovalWorkflow:
    """Test recipe creation and approval workflow with RBAC."""

    def test_engineer_creates_recipe_for_approval(self, engineer_token):
        """Engineer can create a recipe in pending approval status."""
        client = TestClient(lims_app)

        response = client.post(
            "/api/recipes",
            json={
                "name": "New SIMS Recipe",
                "version": "1.0",
                "category": "sims",
                "steps": [
                    {"step": 1, "action": "Set voltage to 5kV"},
                    {"step": 2, "action": "Run scan for 60 minutes"}
                ],
                "approval_required": True
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "PENDING_APPROVAL"
        assert data["name"] == "New SIMS Recipe"

        return data["id"]

    def test_engineer_cannot_approve_own_recipe(self, engineer_token):
        """Engineer cannot approve recipes - requires PI or Admin."""
        client = TestClient(lims_app)

        # Create recipe
        response = client.post(
            "/api/recipes",
            json={
                "name": "Test Recipe",
                "version": "1.0",
                "category": "general",
                "steps": [],
                "approval_required": True
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        recipe_id = response.json()["id"]

        # Try to approve (should fail)
        response = client.post(
            f"/api/recipes/{recipe_id}/approve",
            json={"comments": "Looks good"},
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 403  # Forbidden

    def test_pi_can_approve_recipe(self, engineer_token, pi_token):
        """PI can approve pending recipes."""
        client = TestClient(lims_app)

        # Engineer creates recipe
        response = client.post(
            "/api/recipes",
            json={
                "name": "Recipe Needs Approval",
                "version": "1.0",
                "category": "general",
                "steps": [],
                "approval_required": True
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        recipe_id = response.json()["id"]

        # PI approves recipe
        response = client.post(
            f"/api/recipes/{recipe_id}/approve",
            json={"comments": "Approved by PI"},
            headers={"Authorization": f"Bearer {pi_token['access_token']}"}
        )

        assert response.status_code == 201
        approval_data = response.json()
        assert approval_data["state"] == "APPROVED"

        # Verify recipe status updated
        response = client.get(
            f"/api/recipes/{recipe_id}",
            headers={"Authorization": f"Bearer {pi_token['access_token']}"}
        )

        assert response.json()["status"] == "APPROVED"

    def test_admin_can_approve_recipe(self, engineer_token, admin_token):
        """Admin can approve pending recipes."""
        client = TestClient(lims_app)

        # Engineer creates recipe
        response = client.post(
            "/api/recipes",
            json={
                "name": "Recipe for Admin Approval",
                "version": "1.0",
                "category": "general",
                "steps": [],
                "approval_required": True
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        recipe_id = response.json()["id"]

        # Admin approves recipe
        response = client.post(
            f"/api/recipes/{recipe_id}/approve",
            json={"comments": "Approved by Admin"},
            headers={"Authorization": f"Bearer {admin_token['access_token']}"}
        )

        assert response.status_code == 201
        assert response.json()["state"] == "APPROVED"

    def test_pi_can_reject_recipe(self, engineer_token, pi_token):
        """PI can reject pending recipes."""
        client = TestClient(lims_app)

        # Engineer creates recipe
        response = client.post(
            "/api/recipes",
            json={
                "name": "Recipe to Reject",
                "version": "1.0",
                "category": "general",
                "steps": [],
                "approval_required": True
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        recipe_id = response.json()["id"]

        # PI rejects recipe
        response = client.post(
            f"/api/recipes/{recipe_id}/reject",
            json={"comments": "Needs more steps"},
            headers={"Authorization": f"Bearer {pi_token['access_token']}"}
        )

        assert response.status_code == 201
        approval_data = response.json()
        assert approval_data["state"] == "REJECTED"

        # Verify recipe status updated
        response = client.get(
            f"/api/recipes/{recipe_id}",
            headers={"Authorization": f"Bearer {pi_token['access_token']}"}
        )

        assert response.json()["status"] == "REJECTED"

    def test_cannot_approve_already_approved_recipe(self, engineer_token, pi_token, admin_token):
        """Cannot approve a recipe that's already approved."""
        client = TestClient(lims_app)

        # Engineer creates recipe
        response = client.post(
            "/api/recipes",
            json={
                "name": "Already Approved Recipe",
                "version": "1.0",
                "category": "general",
                "steps": [],
                "approval_required": True
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        recipe_id = response.json()["id"]

        # PI approves
        client.post(
            f"/api/recipes/{recipe_id}/approve",
            json={"comments": "First approval"},
            headers={"Authorization": f"Bearer {pi_token['access_token']}"}
        )

        # Try to approve again (should fail)
        response = client.post(
            f"/api/recipes/{recipe_id}/approve",
            json={"comments": "Second approval attempt"},
            headers={"Authorization": f"Bearer {admin_token['access_token']}"}
        )

        assert response.status_code == 400  # Bad Request
        assert "not pending approval" in response.json()["detail"].lower()

    def test_recipe_without_approval_requirement(self, engineer_token):
        """Recipes can be created without requiring approval."""
        client = TestClient(lims_app)

        response = client.post(
            "/api/recipes",
            json={
                "name": "No Approval Needed",
                "version": "1.0",
                "category": "general",
                "steps": [],
                "approval_required": False
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        assert response.status_code == 201
        data = response.json()

        # Should be in DRAFT status, not PENDING_APPROVAL
        assert data["status"] != "PENDING_APPROVAL"

    def test_viewer_cannot_see_approval_endpoints(self, engineer_token, viewer_token):
        """Viewer (read-only) cannot approve or reject recipes."""
        client = TestClient(lims_app)

        # Engineer creates recipe
        response = client.post(
            "/api/recipes",
            json={
                "name": "Test Recipe",
                "version": "1.0",
                "category": "general",
                "steps": [],
                "approval_required": True
            },
            headers={"Authorization": f"Bearer {engineer_token['access_token']}"}
        )

        recipe_id = response.json()["id"]

        # Viewer tries to approve (should fail)
        response = client.post(
            f"/api/recipes/{recipe_id}/approve",
            json={"comments": "Viewer approval"},
            headers={"Authorization": f"Bearer {viewer_token['access_token']}"}
        )

        assert response.status_code == 403  # Forbidden

        # Viewer tries to reject (should also fail)
        response = client.post(
            f"/api/recipes/{recipe_id}/reject",
            json={"comments": "Viewer rejection"},
            headers={"Authorization": f"Bearer {viewer_token['access_token']}"}
        )

        assert response.status_code == 403  # Forbidden
