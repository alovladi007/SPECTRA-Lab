#!/bin/bash
# deploy_session17.sh
# Complete deployment script for SPECTRA-Lab Platform Session 17
#
# Usage:
#   ./deploy_session17.sh [dev|staging|prod]
#
# Environment variables required:
#   DATABASE_URL - PostgreSQL connection string
#   JWT_SECRET - Secret key for JWT signing (dev only)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check environment
ENV=${1:-dev}
if [[ ! "$ENV" =~ ^(dev|staging|prod)$ ]]; then
    error "Invalid environment. Use: dev, staging, or prod"
fi

log "Deploying SPECTRA-Lab Session 17 to $ENV environment"

# ============================================================================
# Step 1: Prerequisites Check
# ============================================================================

log "Step 1: Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    error "Docker is not installed"
fi

if ! command -v python3 &> /dev/null; then
    error "Python 3 is not installed"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.11"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    error "Python $REQUIRED_VERSION or higher required (found $PYTHON_VERSION)"
fi

success "Prerequisites check passed"

# ============================================================================
# Step 2: Environment Configuration
# ============================================================================

log "Step 2: Configuring environment..."

# Set defaults for dev
if [ "$ENV" = "dev" ]; then
    export DATABASE_URL=${DATABASE_URL:-"postgresql+psycopg://spectra:spectra@localhost:5432/spectra"}
    export JWT_SECRET=${JWT_SECRET:-"dev-secret-change-in-production"}
    export JWT_ALGORITHM=${JWT_ALGORITHM:-"HS256"}
    export OIDC_ENABLED=${OIDC_ENABLED:-"false"}
fi

# Validate required variables
if [ -z "$DATABASE_URL" ]; then
    error "DATABASE_URL environment variable not set"
fi

success "Environment configured for $ENV"

# ============================================================================
# Step 3: Database Setup
# ============================================================================

log "Step 3: Setting up database..."

if [ "$ENV" = "dev" ]; then
    # Start PostgreSQL in Docker
    log "Starting PostgreSQL container..."
    docker compose up -d db
    
    # Wait for PostgreSQL
    log "Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker compose exec -T db pg_isready -U postgres &> /dev/null; then
            success "PostgreSQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            error "PostgreSQL failed to start"
        fi
        sleep 1
    done
fi

# ============================================================================
# Step 4: Python Dependencies
# ============================================================================

log "Step 4: Installing Python dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    success "Created virtual environment"
fi

source venv/bin/activate

# Install dependencies
cat > requirements_session17.txt << 'EOF'
# Core
sqlalchemy>=2.0.0
alembic>=1.12.0
psycopg[binary]>=3.1.0
pydantic>=2.4.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0

# Auth
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6

# Utilities
python-dotenv>=1.0.0
httpx>=0.25.0
redis>=5.0.0
celery>=5.3.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
httpx>=0.25.0
EOF

pip install -r requirements_session17.txt --quiet

if [ $? -eq 0 ]; then
    success "Python dependencies installed"
else
    error "Failed to install Python dependencies"
fi

# ============================================================================
# Step 5: Database Migrations
# ============================================================================

log "Step 5: Running database migrations..."

# Run Alembic migrations
alembic upgrade head

if [ $? -eq 0 ]; then
    success "Database migrations completed"
else
    error "Database migrations failed"
fi

# ============================================================================
# Step 6: Seed Demo Data
# ============================================================================

if [ "$ENV" = "dev" ]; then
    log "Step 6: Seeding demo data..."
    python scripts/seed_demo.py
    
    if [ $? -eq 0 ]; then
        success "Demo data seeded"
    else
        warning "Demo data seeding encountered issues (non-fatal)"
    fi
fi

# ============================================================================
# Step 7: Deploy Services
# ============================================================================

log "Step 7: Deploying services..."

if [ "$ENV" = "dev" ]; then
    # Start all services
    docker compose up -d
    
    # Wait for services
    log "Waiting for services to be ready..."
    sleep 5
    
    # Check service health
    SERVICES=("analysis:8001" "lims:8002" "web:3012")
    for service in "${SERVICES[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -f http://localhost:$port/health &> /dev/null || curl -f http://localhost:$port/ &> /dev/null; then
            success "$name service is healthy"
        else
            warning "$name service may not be ready (check with: docker compose logs $name)"
        fi
    done
    
elif [ "$ENV" = "staging" ] || [ "$ENV" = "prod" ]; then
    log "Deploying to Kubernetes..."
    
    # Apply migrations first
    kubectl apply -f k8s/migration-job.yaml
    kubectl wait --for=condition=complete job/alembic-migration --timeout=300s
    
    # Deploy services
    helm upgrade --install spectra-lab ./helm/spectra-lab \
        --namespace spectra-lab-$ENV \
        --values ./helm/values-$ENV.yaml \
        --wait
    
    success "Kubernetes deployment completed"
fi

# ============================================================================
# Step 8: Validation Tests
# ============================================================================

log "Step 8: Running validation tests..."

if [ "$ENV" = "dev" ]; then
    # Run integration tests
    pytest tests/integration/test_session17.py -v
    
    if [ $? -eq 0 ]; then
        success "Validation tests passed"
    else
        warning "Some validation tests failed"
    fi
fi

# ============================================================================
# Step 9: Display Access Information
# ============================================================================

log "Step 9: Deployment summary..."

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  SPECTRA-Lab Session 17 Deployment Complete"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Environment: $ENV"
echo ""
echo "Services:"
echo "  • Web UI:           http://localhost:3012"
echo "  • Analysis API:     http://localhost:8001"
echo "  • LIMS API:         http://localhost:8002"
echo "  • API Docs:         http://localhost:8001/docs"
echo ""
echo "Database:"
echo "  • PostgreSQL:       localhost:5432/spectra"
echo "  • Run migrations:   alembic upgrade head"
echo ""

if [ "$ENV" = "dev" ]; then
    echo "Demo Credentials:"
    echo "  • Admin:            admin@demo.lab / admin123"
    echo "  • PI:               pi@demo.lab / pi123"
    echo "  • Engineer:         engineer@demo.lab / eng123"
    echo "  • Technician:       tech@demo.lab / tech123"
    echo "  • Viewer:           viewer@demo.lab / view123"
    echo ""
fi

echo "Useful Commands:"
echo "  • View logs:        docker compose logs -f"
echo "  • Restart services: docker compose restart"
echo "  • Stop all:         docker compose down"
echo "  • Database shell:   docker compose exec db psql -U spectra"
echo ""
echo "Next Steps:"
echo "  1. Test login at http://localhost:3012/login"
echo "  2. Create a sample via LIMS API"
echo "  3. Review Session 17 documentation: docs/SESSION_17.md"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

success "Session 17 deployment complete! 🚀"

# ============================================================================
# Cleanup
# ============================================================================

deactivate 2>/dev/null || true
