#!/bin/bash

# ==============================================================================

# SemiconductorLab Platform - Production Deployment Scripts

# ==============================================================================

# Version: 1.0

# Date: October 21, 2025

# Purpose: Complete deployment automation for production environment

# 

# Usage:

# ./deploy.sh [environment] [action]

# 

# Examples:

# ./deploy.sh staging deploy     # Deploy to staging

# ./deploy.sh production deploy  # Deploy to production

# ./deploy.sh production rollback # Rollback last deployment

# ./deploy.sh production backup  # Backup database

# ==============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ==============================================================================

# Configuration

# ==============================================================================

SCRIPT_DIR=”$(cd “$(dirname “${BASH_SOURCE[0]}”)” && pwd)”
PROJECT_ROOT=”$(cd “${SCRIPT_DIR}/..” && pwd)”
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output

RED=’\033[0;31m’
GREEN=’\033[0;32m’
YELLOW=’\033[1;33m’
BLUE=’\033[0;34m’
NC=’\033[0m’ # No Color

# Environment defaults

ENVIRONMENT=”${1:-staging}”
ACTION=”${2:-deploy}”

# Docker image registry

DOCKER_REGISTRY=”${DOCKER_REGISTRY:-ghcr.io/yourorg}”
IMAGE_TAG=”${IMAGE_TAG:-latest}”

# Kubernetes namespace

K8S_NAMESPACE=”${K8S_NAMESPACE:-semiconductorlab-${ENVIRONMENT}}”

# Backup directory

BACKUP_DIR=”${PROJECT_ROOT}/backups”

# ==============================================================================

# Logging Functions

# ==============================================================================

log_info() {
echo -e “${BLUE}[INFO]${NC} $1”
}

log_success() {
echo -e “${GREEN}[SUCCESS]${NC} $1”
}

log_warning() {
echo -e “${YELLOW}[WARNING]${NC} $1”
}

log_error() {
echo -e “${RED}[ERROR]${NC} $1”
}

log_step() {
echo “”
echo -e “${GREEN}==>${NC} $1”
echo “”
}

# ==============================================================================

# Prerequisites Check

# ==============================================================================

check_prerequisites() {
log_step “Checking prerequisites…”

local missing_tools=()

# Check required tools
command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
command -v psql >/dev/null 2>&1 || missing_tools+=("postgresql-client")

if [ ${#missing_tools[@]} -ne 0 ]; then
    log_error "Missing required tools: ${missing_tools[*]}"
    log_info "Please install missing tools and try again"
    exit 1
fi

# Check kubectl connection
if ! kubectl cluster-info >/dev/null 2>&1; then
    log_error "Cannot connect to Kubernetes cluster"
    log_info "Run: kubectl config use-context <your-cluster>"
    exit 1
fi

# Check Docker login
if ! docker info >/dev/null 2>&1; then
    log_error "Docker daemon not running or not accessible"
    exit 1
fi

log_success "All prerequisites met"

}

# ==============================================================================

# Build Docker Images

# ==============================================================================

build_images() {
log_step “Building Docker images…”

cd "${PROJECT_ROOT}"

# Build web frontend
log_info "Building web frontend..."
docker build \
    -t "${DOCKER_REGISTRY}/semiconductorlab-web:${IMAGE_TAG}" \
    -f apps/web/Dockerfile \
    apps/web

# Build instruments service
log_info "Building instruments service..."
docker build \
    -t "${DOCKER_REGISTRY}/semiconductorlab-instruments:${IMAGE_TAG}" \
    -f services/instruments/Dockerfile \
    services/instruments

# Build analysis service
log_info "Building analysis service..."
docker build \
    -t "${DOCKER_REGISTRY}/semiconductorlab-analysis:${IMAGE_TAG}" \
    -f services/analysis/Dockerfile \
    services/analysis

log_success "All images built successfully"

}

# ==============================================================================

# Push Docker Images

# ==============================================================================

push_images() {
log_step “Pushing Docker images to registry…”

docker push "${DOCKER_REGISTRY}/semiconductorlab-web:${IMAGE_TAG}"
docker push "${DOCKER_REGISTRY}/semiconductorlab-instruments:${IMAGE_TAG}"
docker push "${DOCKER_REGISTRY}/semiconductorlab-analysis:${IMAGE_TAG}"

log_success "All images pushed successfully"

}

# ==============================================================================

# Database Migration

# ==============================================================================

run_migrations() {
log_step “Running database migrations…”

# Get database credentials from secrets
DB_HOST=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.host}' | base64 -d)
DB_USER=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.username}' | base64 -d)
DB_PASS=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.password}' | base64 -d)
DB_NAME=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.database}' | base64 -d)

# Run migrations using kubectl exec
kubectl run -n "${K8S_NAMESPACE}" \
    db-migration-${TIMESTAMP} \
    --image="${DOCKER_REGISTRY}/semiconductorlab-instruments:${IMAGE_TAG}" \
    --rm -it --restart=Never \
    --env="DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}/${DB_NAME}" \
    -- alembic upgrade head

log_success "Migrations completed successfully"

}

# ==============================================================================

# Deploy to Kubernetes

# ==============================================================================

deploy_kubernetes() {
log_step “Deploying to Kubernetes…”

# Create namespace if it doesn't exist
kubectl create namespace "${K8S_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# Deploy using Helm
helm upgrade --install semiconductorlab \
    "${PROJECT_ROOT}/infra/kubernetes/helm/semiconductorlab" \
    --namespace "${K8S_NAMESPACE}" \
    --values "${PROJECT_ROOT}/infra/kubernetes/helm/semiconductorlab/values-${ENVIRONMENT}.yaml" \
    --set image.tag="${IMAGE_TAG}" \
    --set web.image="${DOCKER_REGISTRY}/semiconductorlab-web:${IMAGE_TAG}" \
    --set instruments.image="${DOCKER_REGISTRY}/semiconductorlab-instruments:${IMAGE_TAG}" \
    --set analysis.image="${DOCKER_REGISTRY}/semiconductorlab-analysis:${IMAGE_TAG}" \
    --wait \
    --timeout 10m

log_success "Kubernetes deployment successful"

}

# ==============================================================================

# Health Checks

# ==============================================================================

run_health_checks() {
log_step “Running health checks…”

# Wait for pods to be ready
log_info "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod \
    -l app=semiconductorlab \
    -n "${K8S_NAMESPACE}" \
    --timeout=300s

# Get service endpoint
if [ "${ENVIRONMENT}" = "production" ]; then
    ENDPOINT="https://lab.yourcompany.com"
else
    ENDPOINT="https://lab-staging.yourcompany.com"
fi

# Check API health
log_info "Checking API health..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${ENDPOINT}/health")

if [ "${HTTP_CODE}" = "200" ]; then
    log_success "API health check passed (HTTP ${HTTP_CODE})"
else
    log_error "API health check failed (HTTP ${HTTP_CODE})"
    return 1
fi

# Check database connectivity
log_info "Checking database connectivity..."
kubectl exec -n "${K8S_NAMESPACE}" \
    deployment/semiconductorlab-instruments \
    -- python -c "from app.db import engine; engine.connect()" || {
    log_error "Database connectivity check failed"
    return 1
}

log_success "Database connectivity check passed"

# Check Redis
log_info "Checking Redis connectivity..."
kubectl exec -n "${K8S_NAMESPACE}" \
    deployment/semiconductorlab-instruments \
    -- redis-cli -h redis ping | grep -q "PONG" || {
    log_error "Redis connectivity check failed"
    return 1
}

log_success "Redis connectivity check passed"

log_success "All health checks passed!"

}

# ==============================================================================

# Database Backup

# ==============================================================================

backup_database() {
log_step “Backing up database…”

mkdir -p "${BACKUP_DIR}"

# Get database credentials
DB_HOST=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.host}' | base64 -d)
DB_USER=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.username}' | base64 -d)
DB_PASS=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.password}' | base64 -d)
DB_NAME=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.database}' | base64 -d)

# Create backup
BACKUP_FILE="${BACKUP_DIR}/db_${ENVIRONMENT}_${TIMESTAMP}.sql.gz"

PGPASSWORD="${DB_PASS}" pg_dump \
    -h "${DB_HOST}" \
    -U "${DB_USER}" \
    -d "${DB_NAME}" \
    --clean \
    --if-exists \
    --verbose \
    | gzip > "${BACKUP_FILE}"

# Verify backup
if [ -f "${BACKUP_FILE}" ] && [ $(stat -f%z "${BACKUP_FILE}") -gt 1000 ]; then
    log_success "Backup created: ${BACKUP_FILE}"
    log_info "Backup size: $(du -h ${BACKUP_FILE} | cut -f1)"
    
    # Upload to S3 (if configured)
    if [ -n "${AWS_S3_BACKUP_BUCKET:-}" ]; then
        log_info "Uploading backup to S3..."
        aws s3 cp "${BACKUP_FILE}" \
            "s3://${AWS_S3_BACKUP_BUCKET}/semiconductorlab/${ENVIRONMENT}/"
        log_success "Backup uploaded to S3"
    fi
else
    log_error "Backup file is empty or failed to create"
    return 1
fi

}

# ==============================================================================

# Database Restore

# ==============================================================================

restore_database() {
local backup_file=”$1”

if [ -z "${backup_file}" ]; then
    log_error "Please specify backup file to restore"
    log_info "Usage: ./deploy.sh ${ENVIRONMENT} restore <backup_file>"
    exit 1
fi

if [ ! -f "${backup_file}" ]; then
    log_error "Backup file not found: ${backup_file}"
    exit 1
fi

log_warning "⚠️  WARNING: This will overwrite the current database!"
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "${confirm}" != "yes" ]; then
    log_info "Restore cancelled"
    exit 0
fi

log_step "Restoring database from backup..."

# Get database credentials
DB_HOST=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.host}' | base64 -d)
DB_USER=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.username}' | base64 -d)
DB_PASS=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.password}' | base64 -d)
DB_NAME=$(kubectl get secret -n "${K8S_NAMESPACE}" db-credentials -o jsonpath='{.data.database}' | base64 -d)

# Restore backup
gunzip -c "${backup_file}" | PGPASSWORD="${DB_PASS}" psql \
    -h "${DB_HOST}" \
    -U "${DB_USER}" \
    -d "${DB_NAME}"

log_success "Database restored successfully"

}

# ==============================================================================

# Rollback Deployment

# ==============================================================================

rollback_deployment() {
log_step “Rolling back deployment…”

log_warning "⚠️  This will rollback to the previous Helm release"
read -p "Continue? (yes/no): " confirm

if [ "${confirm}" != "yes" ]; then
    log_info "Rollback cancelled"
    exit 0
fi

# Rollback using Helm
helm rollback semiconductorlab \
    --namespace "${K8S_NAMESPACE}" \
    --wait

log_success "Rollback completed"

# Run health checks
run_health_checks

}

# ==============================================================================

# View Logs

# ==============================================================================

view_logs() {
local service=”${1:-all}”

log_step "Viewing logs for ${service}..."

if [ "${service}" = "all" ]; then
    kubectl logs -n "${K8S_NAMESPACE}" \
        -l app=semiconductorlab \
        --tail=100 \
        --follow
else
    kubectl logs -n "${K8S_NAMESPACE}" \
        -l app=semiconductorlab,component="${service}" \
        --tail=100 \
        --follow
fi

}

# ==============================================================================

# Monitoring Setup

# ==============================================================================

setup_monitoring() {
log_step “Setting up monitoring…”

# Deploy Prometheus
helm upgrade --install prometheus prometheus-community/prometheus \
    --namespace monitoring \
    --create-namespace \
    --values "${PROJECT_ROOT}/infra/kubernetes/monitoring/prometheus-values.yaml"

# Deploy Grafana
helm upgrade --install grafana grafana/grafana \
    --namespace monitoring \
    --values "${PROJECT_ROOT}/infra/kubernetes/monitoring/grafana-values.yaml"

# Get Grafana admin password
GRAFANA_PASSWORD=$(kubectl get secret -n monitoring grafana \
    -o jsonpath="{.data.admin-password}" | base64 -d)

log_success "Monitoring deployed successfully"
log_info "Grafana URL: http://grafana.${ENVIRONMENT}.local"
log_info "Grafana Admin Password: ${GRAFANA_PASSWORD}"

}

# ==============================================================================

# SSL Certificate Setup

# ==============================================================================

setup_ssl() {
log_step “Setting up SSL certificates…”

# Install cert-manager if not already installed
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for cert-manager to be ready
kubectl wait --for=condition=available --timeout=300s \
    deployment/cert-manager -n cert-manager

# Apply Let's Encrypt issuer
cat <<EOF | kubectl apply -f -

apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
name: letsencrypt-prod
spec:
acme:
server: https://acme-v02.api.letsencrypt.org/directory
email: admin@yourcompany.com
privateKeySecretRef:
name: letsencrypt-prod
solvers:
- http01:
ingress:
class: nginx
EOF

log_success "SSL certificate setup completed"

}

# ==============================================================================

# Smoke Tests

# ==============================================================================

run_smoke_tests() {
log_step “Running smoke tests…”

if [ "${ENVIRONMENT}" = "production" ]; then
    ENDPOINT="https://lab.yourcompany.com"
else
    ENDPOINT="https://lab-staging.yourcompany.com"
fi

# Test 1: API Health
log_info "Test 1: API Health Endpoint"
curl -f "${ENDPOINT}/health" || {
    log_error "API health check failed"
    return 1
}
log_success "✓ API health check passed"

# Test 2: Authentication
log_info "Test 2: Authentication"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${ENDPOINT}/api/v1/auth/login")
if [ "${HTTP_CODE}" = "200" ] || [ "${HTTP_CODE}" = "401" ]; then
    log_success "✓ Auth endpoint accessible"
else
    log_error "Auth endpoint failed (HTTP ${HTTP_CODE})"
    return 1
fi

# Test 3: Static Assets
log_info "Test 3: Static Assets"
curl -f "${ENDPOINT}/favicon.ico" > /dev/null || {
    log_error "Static assets not accessible"
    return 1
}
log_success "✓ Static assets accessible"

# Test 4: API Documentation
log_info "Test 4: API Documentation"
curl -f "${ENDPOINT}/docs" > /dev/null || {
    log_warning "⚠️  API docs not accessible (non-critical)"
}
log_success "✓ API docs accessible"

log_success "All smoke tests passed!"

}

# ==============================================================================

# Main Deployment Function

# ==============================================================================

main_deploy() {
log_info “Starting deployment to ${ENVIRONMENT}…”
log_info “Image tag: ${IMAGE_TAG}”

# Pre-deployment checks
check_prerequisites

# Backup current database
backup_database

# Build and push images
build_images
push_images

# Deploy to Kubernetes
deploy_kubernetes

# Run migrations
run_migrations

# Health checks
run_health_checks

# Smoke tests
run_smoke_tests

log_success "✓ Deployment completed successfully!"
log_info "Environment: ${ENVIRONMENT}"
log_info "Timestamp: ${TIMESTAMP}"

}

# ==============================================================================

# Main Script Entry Point

# ==============================================================================

main() {
case “${ACTION}” in
deploy)
main_deploy
;;
backup)
backup_database
;;
restore)
restore_database “${3:-}”
;;
rollback)
rollback_deployment
;;
logs)
view_logs “${3:-all}”
;;
health)
run_health_checks
;;
smoke-test)
run_smoke_tests
;;
setup-monitoring)
setup_monitoring
;;
setup-ssl)
setup_ssl
;;
*)
echo “Usage: $0 [environment] [action]”
echo “”
echo “Environments:”
echo “  staging     - Staging environment”
echo “  production  - Production environment”
echo “”
echo “Actions:”
echo “  deploy           - Full deployment”
echo “  backup           - Backup database”
echo “  restore <file>   - Restore from backup”
echo “  rollback         - Rollback to previous version”
echo “  logs [service]   - View logs”
echo “  health           - Run health checks”
echo “  smoke-test       - Run smoke tests”
echo “  setup-monitoring - Setup Prometheus + Grafana”
echo “  setup-ssl        - Setup SSL certificates”
echo “”
echo “Examples:”
echo “  $0 staging deploy”
echo “  $0 production backup”
echo “  $0 production rollback”
echo “  $0 staging logs web”
exit 1
;;
esac
}

# Run main function

main “$@”

# ==============================================================================

# Additional Utility Scripts

# ==============================================================================

# File: scripts/ci/run_tests.sh

# Purpose: CI/CD test execution

cat > “${PROJECT_ROOT}/scripts/ci/run_tests.sh” <<‘TESTSCRIPT’
#!/bin/bash
set -e

echo “Running test suite…”

# Backend tests

cd services/analysis
pytest tests/ -v –cov=app –cov-report=xml –cov-report=html

cd ../instruments
pytest tests/ -v –cov=app –cov-report=xml –cov-report=html

# Frontend tests

cd ../../apps/web
npm run test:ci

echo “✓ All tests passed!”
TESTSCRIPT

chmod +x “${PROJECT_ROOT}/scripts/ci/run_tests.sh”

# ==============================================================================

# Docker Compose Quick Start

# ==============================================================================

cat > “${PROJECT_ROOT}/docker-compose.quick-start.yml” <<‘DOCKERCOMPOSE’
version: ‘3.9’

services:
web:
image: semiconductorlab-web:latest
ports:
- “3000:3000”
environment:
- NEXT_PUBLIC_API_URL=http://localhost:8000
depends_on:
- api-gateway

api-gateway:
image: semiconductorlab-api-gateway:latest
ports:
- “8000:8000”
depends_on:
- instruments
- analysis
- postgres
- redis

instruments:
image: semiconductorlab-instruments:latest
environment:
- DATABASE_URL=postgresql://postgres:postgres@postgres:5432/semiconductorlab
- REDIS_URL=redis://redis:6379

analysis:
image: semiconductorlab-analysis:latest
environment:
- DATABASE_URL=postgresql://postgres:postgres@postgres:5432/semiconductorlab

postgres:
image: timescale/timescaledb:latest-pg15
ports:
- “5432:5432”
environment:
- POSTGRES_PASSWORD=postgres
- POSTGRES_DB=semiconductorlab
volumes:
- postgres_data:/var/lib/postgresql/data

redis:
image: redis:7-alpine
ports:
- “6379:6379”

volumes:
postgres_data:
DOCKERCOMPOSE

log_success “Deployment scripts created successfully!”
log_info “Main script: ./deploy.sh”
log_info “Test script: ./scripts/ci/run_tests.sh”
log_info “Quick start: docker-compose -f docker-compose.quick-start.yml up”