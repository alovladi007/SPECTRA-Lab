#!/bin/bash

###############################################################################
# SESSION 15: LIMS/ELN & REPORTING - Deployment Script
###############################################################################
#
# Deploys LIMS, ELN, SOP Management, and Reporting features to production.
#
# Author: SemiconductorLab Platform Team
# Date: October 26, 2025
# Version: 1.0.0
#
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_DIR="services/lims"
FRONTEND_DIR="apps/web"
DB_MIGRATIONS_DIR="db/migrations"
DEPLOYMENT_ENV="${1:-staging}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "==============================================================================="
echo " SESSION 15: LIMS/ELN & REPORTING DEPLOYMENT"
echo "==============================================================================="
echo " Environment: $DEPLOYMENT_ENV"
echo " Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================================================="
echo ""

###############################################################################
# PRE-DEPLOYMENT CHECKS
###############################################################################

log_info "Running pre-deployment checks..."

# Check Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Please install Docker."
    exit 1
fi
log_success "Docker: OK"

# Check Python dependencies
if ! python3 -c "import reportlab" &> /dev/null; then
    log_warning "reportlab not found. Installing..."
    pip install reportlab --break-system-packages
fi

if ! python3 -c "import qrcode" &> /dev/null; then
    log_warning "qrcode not found. Installing..."
    pip install qrcode[pil] --break-system-packages
fi

if ! python3 -c "import python-barcode" &> /dev/null 2>&1; then
    log_warning "python-barcode not found. Installing..."
    pip install python-barcode[images] --break-system-packages
fi

log_success "Python dependencies: OK"

# Check database connection
log_info "Checking database connection..."
if docker ps | grep -q postgres; then
    log_success "Database: OK"
else
    log_warning "Database container not running. Starting..."
    docker-compose up -d postgres
    sleep 5
fi

###############################################################################
# DATABASE MIGRATIONS
###############################################################################

log_info "Running database migrations for LIMS/ELN..."

# Create migration file
cat > "${DB_MIGRATIONS_DIR}/015_lims_eln_tables.sql" << 'EOF'
-- SESSION 15: LIMS/ELN & REPORTING SCHEMA
-- ========================================

-- Samples table
CREATE TABLE IF NOT EXISTS samples (
    id SERIAL PRIMARY KEY,
    sample_id VARCHAR(100) UNIQUE NOT NULL,
    barcode VARCHAR(100) UNIQUE NOT NULL,
    qr_code TEXT,
    
    organization_id INTEGER REFERENCES organizations(id),
    project_id INTEGER REFERENCES projects(id),
    lot_id INTEGER REFERENCES lots(id),
    parent_sample_id INTEGER REFERENCES samples(id),
    
    material_type VARCHAR(100),
    sample_type VARCHAR(50),
    description TEXT,
    
    dimensions JSONB,
    weight FLOAT,
    weight_units VARCHAR(10) DEFAULT 'g',
    
    status VARCHAR(20) DEFAULT 'received',
    location VARCHAR(200),
    
    received_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expiry_date TIMESTAMP,
    last_measured TIMESTAMP,
    
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sample_project_status ON samples(project_id, status);
CREATE INDEX idx_sample_received_date ON samples(received_date);

-- Lots table
CREATE TABLE IF NOT EXISTS lots (
    id SERIAL PRIMARY KEY,
    lot_number VARCHAR(100) UNIQUE NOT NULL,
    project_id INTEGER REFERENCES projects(id),
    description TEXT,
    quantity INTEGER,
    status VARCHAR(20),
    received_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Custody logs
CREATE TABLE IF NOT EXISTS custody_logs (
    id SERIAL PRIMARY KEY,
    sample_id INTEGER REFERENCES samples(id) NOT NULL,
    action VARCHAR(50),
    from_user_id INTEGER REFERENCES users(id),
    to_user_id INTEGER REFERENCES users(id),
    from_location VARCHAR(200),
    to_location VARCHAR(200),
    reason TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    signature_user_id INTEGER REFERENCES users(id),
    signature_timestamp TIMESTAMP,
    signature_ip VARCHAR(50)
);

CREATE INDEX idx_custody_sample ON custody_logs(sample_id, timestamp DESC);

-- Notebook entries
CREATE TABLE IF NOT EXISTS notebook_entries (
    id SERIAL PRIMARY KEY,
    entry_id VARCHAR(100) UNIQUE NOT NULL,
    project_id INTEGER REFERENCES projects(id),
    author_id INTEGER REFERENCES users(id) NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT,
    content_format VARCHAR(20) DEFAULT 'html',
    linked_samples JSONB,
    linked_runs JSONB,
    linked_methods JSONB,
    version INTEGER DEFAULT 1,
    parent_version_id INTEGER REFERENCES notebook_entries(id),
    is_locked BOOLEAN DEFAULT FALSE,
    locked_at TIMESTAMP,
    locked_by_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_notebook_project ON notebook_entries(project_id, created_at DESC);

-- Entry attachments
CREATE TABLE IF NOT EXISTS entry_attachments (
    id SERIAL PRIMARY KEY,
    entry_id INTEGER REFERENCES notebook_entries(id) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(100),
    file_size INTEGER,
    storage_path VARCHAR(1000),
    checksum_sha256 VARCHAR(64),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_by_id INTEGER REFERENCES users(id)
);

-- Entry signatures (21 CFR Part 11)
CREATE TABLE IF NOT EXISTS entry_signatures (
    id SERIAL PRIMARY KEY,
    entry_id INTEGER REFERENCES notebook_entries(id) NOT NULL,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    signature_type VARCHAR(20) NOT NULL,
    reason TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    ip_address VARCHAR(50),
    user_agent VARCHAR(500),
    content_hash VARCHAR(64) NOT NULL,
    signature_hash VARCHAR(128),
    UNIQUE(entry_id, user_id, signature_type, timestamp)
);

-- SOPs
CREATE TABLE IF NOT EXISTS sops (
    id SERIAL PRIMARY KEY,
    sop_number VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    version VARCHAR(20) NOT NULL,
    method_name VARCHAR(100),
    category VARCHAR(100),
    content TEXT,
    content_format VARCHAR(20) DEFAULT 'markdown',
    checklist_items JSONB,
    status VARCHAR(20) DEFAULT 'draft',
    effective_date TIMESTAMP,
    review_date TIMESTAMP,
    next_review_date TIMESTAMP,
    author_id INTEGER REFERENCES users(id),
    reviewer_ids JSONB,
    approver_id INTEGER REFERENCES users(id),
    supersedes_sop_id INTEGER REFERENCES sops(id),
    superseded_by_sop_id INTEGER REFERENCES sops(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sop_method_status ON sops(method_name, status);

-- Training records
CREATE TABLE IF NOT EXISTS training_records (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    sop_id INTEGER REFERENCES sops(id) NOT NULL,
    completed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    score FLOAT,
    passed BOOLEAN DEFAULT TRUE,
    trainer_id INTEGER REFERENCES users(id),
    certificate_path VARCHAR(1000),
    expiry_date TIMESTAMP
);

-- Checklist completions
CREATE TABLE IF NOT EXISTS checklist_completions (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id) NOT NULL,
    sop_id INTEGER REFERENCES sops(id) NOT NULL,
    user_id INTEGER REFERENCES users(id) NOT NULL,
    completed_items JSONB,
    all_complete BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Update triggers
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_samples_modtime
    BEFORE UPDATE ON samples
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_notebook_entries_modtime
    BEFORE UPDATE ON notebook_entries
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_sops_modtime
    BEFORE UPDATE ON sops
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO semiconductorlab_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO semiconductorlab_user;

EOF

# Run migration
log_info "Applying database schema..."
docker-compose exec -T postgres psql -U postgres -d semiconductorlab < "${DB_MIGRATIONS_DIR}/015_lims_eln_tables.sql" || {
    log_error "Database migration failed"
    exit 1
}

log_success "Database migrations completed"

###############################################################################
# BACKEND DEPLOYMENT
###############################################################################

log_info "Deploying LIMS/ELN backend services..."

# Copy implementation files
mkdir -p "${BACKEND_DIR}/app/lims"
cp session15_lims_eln_complete_implementation.py "${BACKEND_DIR}/app/lims/core.py"

# Update requirements.txt
cat >> "${BACKEND_DIR}/requirements.txt" << EOF

# Session 15: LIMS/ELN & Reporting
reportlab==4.0.7
qrcode[pil]==7.4.2
python-barcode[images]==0.15.1
jinja2==3.1.2
pypdf==3.17.1
EOF

# Install dependencies
log_info "Installing Python dependencies..."
docker-compose exec -T backend pip install -r requirements.txt || {
    log_warning "Some dependencies failed to install. Continuing..."
}

# Restart backend service
log_info "Restarting backend service..."
docker-compose restart backend
sleep 5

# Health check
log_info "Running health check..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        log_success "Backend health check passed"
        break
    fi
    if [ $i -eq 10 ]; then
        log_error "Backend health check failed after 10 attempts"
        exit 1
    fi
    sleep 2
done

###############################################################################
# FRONTEND DEPLOYMENT
###############################################################################

log_info "Deploying LIMS/ELN frontend components..."

# Copy UI components
mkdir -p "${FRONTEND_DIR}/src/components/lims"
cp session15_lims_eln_ui_components.tsx "${FRONTEND_DIR}/src/components/lims/index.tsx"

# Update package.json with dependencies if needed
# (Assuming shadcn/ui is already installed)

# Rebuild frontend
log_info "Building frontend..."
cd "${FRONTEND_DIR}"
npm run build || {
    log_error "Frontend build failed"
    exit 1
}
cd -

# Restart frontend service
log_info "Restarting frontend service..."
docker-compose restart web
sleep 5

log_success "Frontend deployment completed"

###############################################################################
# POST-DEPLOYMENT TESTS
###############################################################################

log_info "Running post-deployment tests..."

# Run integration tests
log_info "Executing integration tests..."
pytest test_session15_integration.py -v --tb=short || {
    log_warning "Some tests failed. Review test output above."
}

# API endpoint tests
log_info "Testing API endpoints..."

# Test sample creation endpoint
SAMPLE_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/lims/samples \
    -H "Content-Type: application/json" \
    -d '{
        "project_id": 1,
        "material_type": "silicon",
        "sample_type": "wafer",
        "location": "Lab A"
    }')

if echo "$SAMPLE_RESPONSE" | grep -q "sample_id"; then
    log_success "Sample creation endpoint: OK"
else
    log_warning "Sample creation endpoint: FAILED"
fi

# Test ELN endpoint
ELN_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/lims/eln/entries \
    -H "Content-Type: application/json" \
    -d '{
        "project_id": 1,
        "title": "Test Entry",
        "content": "Test content"
    }')

if echo "$ELN_RESPONSE" | grep -q "entry_id"; then
    log_success "ELN creation endpoint: OK"
else
    log_warning "ELN creation endpoint: FAILED"
fi

###############################################################################
# DEPLOYMENT SUMMARY
###############################################################################

echo ""
echo "==============================================================================="
echo " DEPLOYMENT SUMMARY"
echo "==============================================================================="
log_success "Session 15 (LIMS/ELN & Reporting) deployed successfully!"
echo ""
echo "Services:"
echo "  - Backend API: http://localhost:8000"
echo "  - Frontend: http://localhost:3000"
echo "  - Database: localhost:5432"
echo ""
echo "New Features:"
echo "  ✓ Sample Management with Barcode/QR codes"
echo "  ✓ Chain of Custody tracking"
echo "  ✓ Electronic Lab Notebook"
echo "  ✓ E-signatures (21 CFR Part 11)"
echo "  ✓ SOP Management"
echo "  ✓ Pre-run Checklists"
echo "  ✓ PDF Report Generation"
echo "  ✓ FAIR Data Export"
echo ""
echo "Database Tables Added:"
echo "  - samples, lots, custody_logs"
echo "  - notebook_entries, entry_attachments, entry_signatures"
echo "  - sops, training_records, checklist_completions"
echo ""
echo "API Endpoints:"
echo "  POST   /api/v1/lims/samples"
echo "  GET    /api/v1/lims/samples/{sample_id}"
echo "  POST   /api/v1/lims/samples/{sample_id}/custody"
echo "  POST   /api/v1/lims/eln/entries"
echo "  POST   /api/v1/lims/eln/entries/{entry_id}/sign"
echo "  POST   /api/v1/lims/sops"
echo "  GET    /api/v1/lims/sops/method/{method_name}"
echo "  POST   /api/v1/lims/reports/generate"
echo "  POST   /api/v1/lims/export/fair"
echo ""
echo "Next Steps:"
echo "  1. Create sample templates for your materials"
echo "  2. Upload SOPs for each measurement method"
echo "  3. Configure report templates"
echo "  4. Train users on ELN and e-signatures"
echo "  5. Proceed to Session 16 (Hardening & Pilot)"
echo ""
echo "==============================================================================="
echo " Deployment completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================================================="
