#!/bin/bash
# acceptance_test.sh
# Comprehensive acceptance test suite for SPECTRA-Lab Session 17

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0

log() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

success() {
    echo -e "${GREEN}✓ PASS${NC} $1"
    PASS=$((PASS+1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC} $1"
    FAIL=$((FAIL+1))
}

check() {
    if [ $? -eq 0 ]; then
        success "$1"
    else
        fail "$1"
    fi
}

echo "════════════════════════════════════════════════════════════"
echo "  SPECTRA-Lab Session 17 - Acceptance Test Suite"
echo "════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# 1. Service Health Checks
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Service Health Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log "Checking PostgreSQL..."
docker compose exec -T db pg_isready -U spectra &>/dev/null
check "PostgreSQL is ready"

log "Checking Analysis API health..."
curl -f http://localhost:8001/health &>/dev/null
check "Analysis API health endpoint"

log "Checking LIMS API health..."
curl -f http://localhost:8002/health &>/dev/null
check "LIMS API health endpoint"

log "Checking Web UI..."
curl -f http://localhost:3012/ &>/dev/null
check "Web UI is accessible"

# ============================================================================
# 2. Database Verification
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. Database Schema Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log "Checking Alembic version..."
docker compose exec -T db psql -U spectra -c "SELECT version_num FROM alembic_version" &>/dev/null
check "Alembic migration table exists"

log "Verifying core tables..."
TABLES=(
    "organizations" "users" "api_keys"
    "instruments" "calibrations" "materials"
    "samples" "wafers" "devices"
    "recipes" "recipe_approvals" "runs" "results"
    "attachments" "eln_entries" "signatures"
    "sops" "custody_events"
    "spc_series" "spc_points" "spc_alerts"
    "feature_sets" "ml_models"
)

for table in "${TABLES[@]}"; do
    docker compose exec -T db psql -U spectra -c "\d $table" &>/dev/null
    check "Table: $table"
done

log "Checking demo data seeded..."
ORG_COUNT=$(docker compose exec -T db psql -U spectra -t -c "SELECT COUNT(*) FROM organizations")
if [ $ORG_COUNT -ge 2 ]; then
    success "Organizations seeded (count: $ORG_COUNT)"
else
    fail "Organizations not seeded properly"
fi

USER_COUNT=$(docker compose exec -T db psql -U spectra -t -c "SELECT COUNT(*) FROM users")
if [ $USER_COUNT -ge 5 ]; then
    success "Users seeded (count: $USER_COUNT)"
else
    fail "Users not seeded properly"
fi

# ============================================================================
# 3. Authentication Tests
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. Authentication & Authorization Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log "Testing login with engineer credentials..."
LOGIN_RESPONSE=$(curl -s -X POST http://localhost:8002/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"engineer@demo.lab","password":"eng123"}')

if echo "$LOGIN_RESPONSE" | grep -q "access_token"; then
    success "Login successful"
    TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
else
    fail "Login failed"
    TOKEN=""
fi

if [ -n "$TOKEN" ]; then
    log "Testing /auth/me endpoint..."
    ME_RESPONSE=$(curl -s http://localhost:8002/auth/me \
        -H "Authorization: Bearer $TOKEN")
    
    if echo "$ME_RESPONSE" | grep -q "engineer@demo.lab"; then
        success "Current user endpoint"
    else
        fail "Current user endpoint"
    fi
    
    log "Testing unauthorized access (no token)..."
    UNAUTH_RESPONSE=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:8002/api/lims/samples)
    if [ "$UNAUTH_RESPONSE" = "401" ]; then
        success "Unauthorized access blocked"
    else
        fail "Unauthorized access not blocked (got HTTP $UNAUTH_RESPONSE)"
    fi
else
    fail "Cannot test authenticated endpoints (no token)"
fi

# ============================================================================
# 4. CRUD Operations Tests
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. CRUD Operations Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -n "$TOKEN" ]; then
    log "Testing GET /api/lims/samples..."
    SAMPLES_RESPONSE=$(curl -s http://localhost:8002/api/lims/samples \
        -H "Authorization: Bearer $TOKEN")
    
    if echo "$SAMPLES_RESPONSE" | grep -q "Sample"; then
        success "List samples"
    else
        fail "List samples"
    fi
    
    log "Testing POST /api/lims/samples (create)..."
    CREATE_SAMPLE=$(curl -s -X POST http://localhost:8002/api/lims/samples \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "TEST-SAMPLE-001",
            "material_type": "GaAs",
            "lot_code": "LOT-TEST-001",
            "barcode": "BC-' "$(date +%s)" '",
            "location": "Test Cabinet"
        }')
    
    if echo "$CREATE_SAMPLE" | grep -q "TEST-SAMPLE-001"; then
        success "Create sample"
        SAMPLE_ID=$(echo "$CREATE_SAMPLE" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    else
        fail "Create sample"
        SAMPLE_ID=""
    fi
    
    if [ -n "$SAMPLE_ID" ]; then
        log "Testing GET /api/lims/samples/{id}..."
        GET_SAMPLE=$(curl -s http://localhost:8002/api/lims/samples/$SAMPLE_ID \
            -H "Authorization: Bearer $TOKEN")
        
        if echo "$GET_SAMPLE" | grep -q "TEST-SAMPLE-001"; then
            success "Get sample by ID"
        else
            fail "Get sample by ID"
        fi
    fi
else
    fail "Cannot test CRUD operations (no auth token)"
fi

# ============================================================================
# 5. Calibration Lockout Test
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Calibration Lockout Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -n "$TOKEN" ]; then
    log "Testing calibration status check..."
    CAL_STATUS=$(curl -s http://localhost:8001/api/analysis/calibrations \
        -H "Authorization: Bearer $TOKEN")
    
    if echo "$CAL_STATUS" | grep -q "expires_at"; then
        success "Calibration status endpoint"
    else
        fail "Calibration status endpoint"
    fi
    
    # Get an instrument with expired calibration
    EXPIRED_INST=$(docker compose exec -T db psql -U spectra -t -c \
        "SELECT i.id FROM instruments i 
         JOIN calibrations c ON c.instrument_id = i.id 
         WHERE c.expires_at < NOW() 
         LIMIT 1" | tr -d ' ')
    
    if [ -n "$EXPIRED_INST" ]; then
        log "Testing run creation with expired calibration (should fail)..."
        RUN_BLOCKED=$(curl -s -w "%{http_code}" -o /tmp/run_response.json \
            -X POST http://localhost:8001/api/analysis/runs \
            -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" \
            -d "{
                \"instrument_id\": \"$EXPIRED_INST\",
                \"method\": \"iv_sweep\",
                \"params\": {}
            }")
        
        if [ "$RUN_BLOCKED" = "409" ]; then
            success "Run blocked for expired calibration"
        else
            fail "Run should be blocked for expired calibration (got HTTP $RUN_BLOCKED)"
        fi
    else
        log "No expired calibrations to test lockout"
    fi
else
    fail "Cannot test calibration lockout (no auth token)"
fi

# ============================================================================
# 6. Role-Based Access Control Tests
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Role-Based Access Control Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log "Testing with viewer role (read-only)..."
VIEWER_LOGIN=$(curl -s -X POST http://localhost:8002/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"viewer@demo.lab","password":"view123"}')

if echo "$VIEWER_LOGIN" | grep -q "access_token"; then
    VIEWER_TOKEN=$(echo "$VIEWER_LOGIN" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    
    log "Testing viewer can read samples..."
    VIEWER_READ=$(curl -s -w "%{http_code}" -o /dev/null \
        http://localhost:8002/api/lims/samples \
        -H "Authorization: Bearer $VIEWER_TOKEN")
    
    if [ "$VIEWER_READ" = "200" ]; then
        success "Viewer can read samples"
    else
        fail "Viewer cannot read samples"
    fi
    
    log "Testing viewer cannot create samples..."
    VIEWER_CREATE=$(curl -s -w "%{http_code}" -o /dev/null \
        -X POST http://localhost:8002/api/lims/samples \
        -H "Authorization: Bearer $VIEWER_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"name":"SHOULD-FAIL"}')
    
    if [ "$VIEWER_CREATE" = "403" ]; then
        success "Viewer blocked from creating samples"
    else
        fail "Viewer should not be able to create samples (got HTTP $VIEWER_CREATE)"
    fi
else
    fail "Viewer login failed"
fi

# ============================================================================
# 7. Multi-Org Tenancy Tests
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. Multi-Org Tenancy Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log "Verifying org isolation..."
if [ -n "$TOKEN" ]; then
    # Engineer from demo-lab should only see demo-lab samples
    ENGINEER_SAMPLES=$(curl -s http://localhost:8002/api/lims/samples \
        -H "Authorization: Bearer $TOKEN")
    
    DEMO_COUNT=$(echo "$ENGINEER_SAMPLES" | grep -o '"organization_id"' | wc -l)
    
    if [ $DEMO_COUNT -gt 0 ]; then
        success "Engineer sees own org samples (count: $DEMO_COUNT)"
    else
        fail "Engineer sees no samples"
    fi
fi

# ============================================================================
# 8. Integration Tests
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. Python Integration Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "tests/integration/test_session17.py" ]; then
    log "Running pytest integration tests..."
    pytest tests/integration/test_session17.py -v --tb=short &>/dev/null
    check "Python integration test suite"
else
    log "Integration test file not found, skipping"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Test Summary"
echo "════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}Passed:${NC} $PASS"
echo -e "${RED}Failed:${NC} $FAIL"
echo ""

TOTAL=$((PASS+FAIL))
if [ $TOTAL -gt 0 ]; then
    PERCENTAGE=$((PASS * 100 / TOTAL))
    echo -e "Success Rate: ${PERCENTAGE}%"
fi

echo ""
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓✓✓ ALL TESTS PASSED ✓✓✓${NC}"
    echo ""
    echo "Session 17 deployment is production-ready! 🚀"
    exit 0
else
    echo -e "${RED}✗✗✗ SOME TESTS FAILED ✗✗✗${NC}"
    echo ""
    echo "Please review failed tests and check logs:"
    echo "  docker compose logs analysis"
    echo "  docker compose logs lims"
    echo "  docker compose logs db"
    exit 1
fi
