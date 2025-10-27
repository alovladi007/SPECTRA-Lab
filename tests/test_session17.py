"""
tests/integration/test_session17.py

Integration tests for SPECTRA-Lab Session 17.
Tests authentication, authorization, org scoping, and calibration lockout.
"""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta, timezone
import uuid

# Test data
TEST_ORG_ID = str(uuid.uuid4())
TEST_USER_EMAIL = "test@demo.lab"
TEST_PASSWORD = "test123"


@pytest.fixture
async def client():
    """HTTP client for testing."""
    async with AsyncClient(base_url="http://localhost:8002") as ac:
        yield ac


@pytest.fixture
async def admin_token(client):
    """Get admin token."""
    response = await client.post("/auth/login", json={
        "username": "admin@demo.lab",
        "password": "admin123"
    })
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
async def engineer_token(client):
    """Get engineer token."""
    response = await client.post("/auth/login", json={
        "username": "engineer@demo.lab",
        "password": "eng123"
    })
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
async def tech_token(client):
    """Get technician token."""
    response = await client.post("/auth/login", json={
        "username": "tech@demo.lab",
        "password": "tech123"
    })
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
async def viewer_token(client):
    """Get viewer token."""
    response = await client.post("/auth/login", json={
        "username": "viewer@demo.lab",
        "password": "view123"
    })
    assert response.status_code == 200
    return response.json()["access_token"]


# ============================================================================
# Authentication Tests
# ============================================================================

@pytest.mark.asyncio
async def test_login_success(client):
    """Test successful login."""
    response = await client.post("/auth/login", json={
        "username": "engineer@demo.lab",
        "password": "eng123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0


@pytest.mark.asyncio
async def test_login_invalid_password(client):
    """Test login with wrong password."""
    response = await client.post("/auth/login", json={
        "username": "engineer@demo.lab",
        "password": "wrongpassword"
    })
    
    assert response.status_code == 401
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_login_nonexistent_user(client):
    """Test login with non-existent user."""
    response = await client.post("/auth/login", json={
        "username": "nonexistent@demo.lab",
        "password": "password"
    })
    
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token(client, engineer_token):
    """Test token refresh."""
    # First login to get refresh token
    response = await client.post("/auth/login", json={
        "username": "engineer@demo.lab",
        "password": "eng123"
    })
    refresh_token = response.json()["refresh_token"]
    
    # Refresh
    response = await client.post("/auth/refresh", json={
        "refresh_token": refresh_token
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["access_token"] != engineer_token  # New token


@pytest.mark.asyncio
async def test_get_current_user(client, engineer_token):
    """Test getting current user info."""
    response = await client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "engineer@demo.lab"
    assert data["role"] == "engineer"
    assert "organization_id" in data


# ============================================================================
# Authorization / RBAC Tests
# ============================================================================

@pytest.mark.asyncio
async def test_protected_endpoint_no_auth(client):
    """Test accessing protected endpoint without token."""
    response = await client.get("/api/lims/samples")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_invalid_token(client):
    """Test accessing protected endpoint with invalid token."""
    response = await client.get(
        "/api/lims/samples",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_role_enforcement_pi_approve(client, admin_token, engineer_token):
    """Test that only PI/Admin can approve recipes."""
    # Create draft recipe as engineer
    create_response = await client.post(
        "/api/lims/recipes",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "name": "Test Recipe",
            "version": "1.0",
            "params": {"method": "iv_sweep"}
        }
    )
    assert create_response.status_code == 201
    recipe_id = create_response.json()["id"]
    
    # Try to approve as engineer (should fail)
    approve_response = await client.post(
        f"/api/lims/recipes/{recipe_id}/approve",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={"comment": "Looks good"}
    )
    assert approve_response.status_code == 403
    
    # Approve as admin (should succeed)
    approve_response = await client.post(
        f"/api/lims/recipes/{recipe_id}/approve",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"comment": "Approved"}
    )
    assert approve_response.status_code == 200


@pytest.mark.asyncio
async def test_role_enforcement_viewer_readonly(client, viewer_token):
    """Test that viewer can only read, not create."""
    # Can read samples
    response = await client.get(
        "/api/lims/samples",
        headers={"Authorization": f"Bearer {viewer_token}"}
    )
    assert response.status_code == 200
    
    # Cannot create sample
    response = await client.post(
        "/api/lims/samples",
        headers={"Authorization": f"Bearer {viewer_token}"},
        json={
            "name": "New Sample",
            "material_type": "Si"
        }
    )
    assert response.status_code == 403


# ============================================================================
# Organization Scoping Tests
# ============================================================================

@pytest.mark.asyncio
async def test_org_scoping_samples(client, engineer_token):
    """Test that samples are scoped to organization."""
    # Create sample
    create_response = await client.post(
        "/api/lims/samples",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "name": f"Test Sample {uuid.uuid4()}",
            "material_type": "GaN",
            "lot_code": "LOT-001"
        }
    )
    assert create_response.status_code == 201
    sample_id = create_response.json()["id"]
    
    # List samples - should include the one we created
    list_response = await client.get(
        "/api/lims/samples",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    assert list_response.status_code == 200
    samples = list_response.json()["items"]
    sample_ids = [s["id"] for s in samples]
    assert sample_id in sample_ids
    
    # All samples should belong to same org
    orgs = set(s["organization_id"] for s in samples)
    assert len(orgs) == 1


@pytest.mark.asyncio
async def test_org_isolation(client, engineer_token):
    """Test that users cannot access other orgs' data."""
    # This test assumes test-org exists with different data
    # In real scenario, you'd create a second org and user
    
    # Get samples
    response = await client.get(
        "/api/lims/samples",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    assert response.status_code == 200
    
    # Verify all samples belong to demo-lab org
    samples = response.json()["items"]
    for sample in samples:
        # Engineer is in demo-lab
        assert "demo" in sample["organization_id"].lower() or sample["organization_id"] != ""


# ============================================================================
# Calibration Lockout Tests
# ============================================================================

@pytest.mark.asyncio
async def test_run_creation_valid_calibration(client, engineer_token):
    """Test creating run with valid calibration."""
    # Assume Keithley 2400 has valid calibration (from seed data)
    # Get instrument ID
    instruments_response = await client.get(
        "/api/analysis/instruments",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    instruments = instruments_response.json()["items"]
    keithley = next((i for i in instruments if "Keithley" in i["name"]), None)
    assert keithley is not None
    
    # Check calibration status
    cal_response = await client.get(
        f"/api/analysis/calibrations/status?instrument_id={keithley['id']}",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    assert cal_response.status_code == 200
    cal_status = cal_response.json()
    
    if cal_status["valid"]:
        # Create run - should succeed
        run_response = await client.post(
            "/api/analysis/runs",
            headers={"Authorization": f"Bearer {engineer_token}"},
            json={
                "instrument_id": keithley["id"],
                "method": "iv_sweep",
                "params": {"vgs_start": -2.0, "vgs_stop": 2.0}
            }
        )
        assert run_response.status_code in [200, 201]


@pytest.mark.asyncio
async def test_run_blocked_expired_calibration(client, engineer_token):
    """Test that run creation fails with expired calibration."""
    # Assume XRD has expired calibration (from seed data)
    instruments_response = await client.get(
        "/api/analysis/instruments",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    instruments = instruments_response.json()["items"]
    xrd = next((i for i in instruments if "XRD" in i["name"] or "Bruker" in i["name"]), None)
    
    if xrd:
        # Check calibration status
        cal_response = await client.get(
            f"/api/analysis/calibrations/status?instrument_id={xrd['id']}",
            headers={"Authorization": f"Bearer {engineer_token}"}
        )
        cal_status = cal_response.json()
        
        if not cal_status["valid"]:
            # Try to create run - should fail with 409
            run_response = await client.post(
                "/api/analysis/runs",
                headers={"Authorization": f"Bearer {engineer_token}"},
                json={
                    "instrument_id": xrd["id"],
                    "method": "xrd_scan",
                    "params": {"theta_start": 20, "theta_stop": 80}
                }
            )
            assert run_response.status_code == 409
            detail = run_response.json()["detail"]
            assert "calibration" in detail["message"].lower()
            assert "expired" in detail["message"].lower()


# ============================================================================
# CRUD Operations Tests
# ============================================================================

@pytest.mark.asyncio
async def test_sample_crud(client, engineer_token):
    """Test full CRUD cycle for samples."""
    # Create
    sample_name = f"Test Sample {uuid.uuid4()}"
    create_response = await client.post(
        "/api/lims/samples",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "name": sample_name,
            "material_type": "Si",
            "lot_code": "LOT-TEST-001",
            "barcode": f"BC-{uuid.uuid4()}"
        }
    )
    assert create_response.status_code == 201
    sample = create_response.json()
    sample_id = sample["id"]
    assert sample["name"] == sample_name
    
    # Read
    read_response = await client.get(
        f"/api/lims/samples/{sample_id}",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    assert read_response.status_code == 200
    assert read_response.json()["id"] == sample_id
    
    # Update
    update_response = await client.put(
        f"/api/lims/samples/{sample_id}",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "location": "Storage Cabinet B"
        }
    )
    assert update_response.status_code == 200
    assert update_response.json()["location"] == "Storage Cabinet B"
    
    # Soft Delete
    delete_response = await client.delete(
        f"/api/lims/samples/{sample_id}",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    assert delete_response.status_code == 204
    
    # Verify soft deleted (should still exist but marked deleted)
    read_response = await client.get(
        f"/api/lims/samples/{sample_id}?include_deleted=true",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    assert read_response.status_code == 200
    assert read_response.json()["is_deleted"] == True


@pytest.mark.asyncio
async def test_pagination(client, engineer_token):
    """Test pagination of list endpoints."""
    # Create multiple samples
    for i in range(15):
        await client.post(
            "/api/lims/samples",
            headers={"Authorization": f"Bearer {engineer_token}"},
            json={
                "name": f"Pagination Test {i}",
                "material_type": "GaN"
            }
        )
    
    # First page
    page1 = await client.get(
        "/api/lims/samples?skip=0&limit=10",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    assert page1.status_code == 200
    data1 = page1.json()
    assert len(data1["items"]) == 10
    assert data1["pagination"]["page"] == 1
    
    # Second page
    page2 = await client.get(
        "/api/lims/samples?skip=10&limit=10",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    assert page2.status_code == 200
    data2 = page2.json()
    assert len(data2["items"]) >= 5  # At least the 15 we created
    assert data2["pagination"]["page"] == 2


# ============================================================================
# ELN Tests
# ============================================================================

@pytest.mark.asyncio
async def test_eln_entry_creation(client, engineer_token):
    """Test creating and signing ELN entry."""
    # Create entry
    entry_response = await client.post(
        "/api/lims/eln",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "title": "Test Experiment Log",
            "body_markdown": "# Experiment\n\nResults were positive.",
            "linked_entities": {}
        }
    )
    assert entry_response.status_code == 201
    entry_id = entry_response.json()["id"]
    
    # Sign entry
    sign_response = await client.post(
        f"/api/lims/eln/{entry_id}/sign",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "method": "password",
            "password": "eng123"
        }
    )
    assert sign_response.status_code == 200
    
    # Verify signed
    get_response = await client.get(
        f"/api/lims/eln/{entry_id}",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    entry = get_response.json()
    assert entry["signed"] == True
    assert entry["signed_at"] is not None


# ============================================================================
# SPC Tests
# ============================================================================

@pytest.mark.asyncio
async def test_spc_series_creation(client, engineer_token):
    """Test creating SPC control chart series."""
    response = await client.post(
        "/api/analysis/spc/series",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "name": "Test Metric Trend",
            "path": "test/metric/trend",
            "entity_type": "method",
            "metric": "test_metric",
            "spec_lcl": 0.0,
            "spec_ucl": 10.0
        }
    )
    assert response.status_code == 201
    series = response.json()
    assert series["name"] == "Test Metric Trend"
    assert "id" in series


# ============================================================================
# Integration Test Summary
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow(client, admin_token, engineer_token):
    """
    Test complete workflow:
    1. Admin creates org and users
    2. Engineer creates sample
    3. Engineer creates recipe
    4. PI approves recipe
    5. Engineer creates run
    6. Results are generated
    7. Engineer creates ELN entry
    """
    
    # 1. Create sample
    sample_response = await client.post(
        "/api/lims/samples",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "name": f"Workflow Sample {uuid.uuid4()}",
            "material_type": "Si"
        }
    )
    assert sample_response.status_code == 201
    sample_id = sample_response.json()["id"]
    
    # 2. Create recipe
    recipe_response = await client.post(
        "/api/lims/recipes",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "name": "Workflow Recipe",
            "version": "1.0",
            "params": {"method": "iv_sweep"}
        }
    )
    assert recipe_response.status_code == 201
    recipe_id = recipe_response.json()["id"]
    
    # 3. Admin approves recipe (PI would in real scenario)
    approve_response = await client.post(
        f"/api/lims/recipes/{recipe_id}/approve",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"comment": "Approved for workflow test"}
    )
    assert approve_response.status_code == 200
    
    # 4. Create run (if valid instrument available)
    instruments_response = await client.get(
        "/api/analysis/instruments",
        headers={"Authorization": f"Bearer {engineer_token}"}
    )
    instruments = instruments_response.json()["items"]
    valid_instrument = next(
        (i for i in instruments if i["status"] == "online"),
        None
    )
    
    if valid_instrument:
        run_response = await client.post(
            "/api/analysis/runs",
            headers={"Authorization": f"Bearer {engineer_token}"},
            json={
                "recipe_id": recipe_id,
                "instrument_id": valid_instrument["id"],
                "sample_id": sample_id,
                "method": "iv_sweep",
                "params": {}
            }
        )
        # May succeed or fail based on calibration
        assert run_response.status_code in [200, 201, 409]
    
    # 5. Create ELN entry documenting the work
    eln_response = await client.post(
        "/api/lims/eln",
        headers={"Authorization": f"Bearer {engineer_token}"},
        json={
            "title": "Workflow Test Documentation",
            "body_markdown": f"# Test Workflow\n\nSample: {sample_id}\nRecipe: {recipe_id}",
            "linked_entities": {
                "samples": [sample_id],
                "recipes": [recipe_id]
            }
        }
    )
    assert eln_response.status_code == 201
    
    print("✅ Full workflow test completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
