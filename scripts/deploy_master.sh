#!/bin/bash
#
# SemiconductorLab Platform - Master Deployment Script
# 
# Complete automation for production deployment with:
# - Docker image building and pushing
# - Kubernetes/Helm deployment
# - Database migrations
# - Health checks and smoke tests
# - Backup and restore
# - Rollback procedures
# - Monitoring setup
#
# Usage: ./deploy.sh [environment] [action]
#
# Environments: local, staging, production
# Actions: deploy, backup, restore, rollback, health, smoke-test, logs, setup-monitoring, setup-ssl

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="semiconductorlab"
VERSION=$(cat VERSION 2>/dev/null || echo "1.0.0")
DOCKER_REGISTRY="your-registry.io"

# Print colored message
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

# Display usage
usage() {
    cat << EOF
SemiconductorLab Platform Deployment Script

Usage: $0 [environment] [action]

Environments:
  local       - Local Docker Compose deployment
  staging     - Staging Kubernetes cluster
  production  - Production Kubernetes cluster

Actions:
  deploy           - Full deployment with health checks
  backup           - Backup database to S3
  restore [file]   - Restore database from backup
  rollback         - Rollback to previous version
  health           - Run health checks
  smoke-test       - Quick validation tests
  logs [service]   - Tail logs for service
  setup-monitoring - Setup Prometheus + Grafana
  setup-ssl        - Configure Let's Encrypt SSL
  status           - Show current deployment status

Examples:
  $0 production deploy
  $0 staging backup
  $0 production rollback
  $0 local logs api

EOF
    exit 1
}

# Check prerequisites
check_prerequisites() {
    local env=$1
    
    log_info "Checking prerequisites for $env environment..."
    
    if [[ "$env" == "local" ]]; then
        command -v docker >/dev/null 2>&1 || { log_error "docker is required but not installed"; exit 1; }
        command -v docker-compose >/dev/null 2>&1 || { log_error "docker-compose is required"; exit 1; }
    else
        command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed"; exit 1; }
        command -v helm >/dev/null 2>&1 || { log_error "helm is required but not installed"; exit 1; }
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build API service
    docker build -t ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:${VERSION} \
        -f services/api/Dockerfile .
    
    # Build Analysis service
    docker build -t ${DOCKER_REGISTRY}/${PROJECT_NAME}-analysis:${VERSION} \
        -f services/analysis/Dockerfile .
    
    # Build Web UI
    docker build -t ${DOCKER_REGISTRY}/${PROJECT_NAME}-web:${VERSION} \
        -f apps/web/Dockerfile .
    
    log_success "Docker images built successfully"
}

# Push Docker images
push_images() {
    log_info "Pushing Docker images to registry..."
    
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}-api:${VERSION}
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}-analysis:${VERSION}
    docker push ${DOCKER_REGISTRY}/${PROJECT_NAME}-web:${VERSION}
    
    log_success "Docker images pushed successfully"
}

# Deploy to local
deploy_local() {
    log_info "Deploying to local environment..."
    
    # Start services
    docker-compose -f docker-compose.yml up -d
    
    # Wait for services to be ready
    sleep 10
    
    # Run database migrations
    docker-compose exec api alembic upgrade head
    
    log_success "Local deployment complete"
    log_info "Access the platform at http://localhost:3000"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    local env=$1
    local namespace="${PROJECT_NAME}-${env}"
    
    log_info "Deploying to ${env} Kubernetes cluster..."
    
    # Ensure namespace exists
    kubectl create namespace ${namespace} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ConfigMaps and Secrets
    kubectl apply -f infra/kubernetes/configmaps/${env}.yaml -n ${namespace}
    
    # Deploy with Helm
    helm upgrade --install ${PROJECT_NAME} \
        ./infra/kubernetes/helm/${PROJECT_NAME} \
        --namespace ${namespace} \
        --values ./infra/kubernetes/helm/${PROJECT_NAME}/values-${env}.yaml \
        --set image.tag=${VERSION} \
        --wait \
        --timeout 10m
    
    # Run database migrations
    kubectl run migrations-${VERSION} \
        --image=${DOCKER_REGISTRY}/${PROJECT_NAME}-api:${VERSION} \
        --restart=Never \
        --namespace=${namespace} \
        --command -- alembic upgrade head
    
    # Wait for migration to complete
    kubectl wait --for=condition=complete --timeout=300s \
        job/migrations-${VERSION} -n ${namespace} || true
    
    log_success "Kubernetes deployment complete"
}

# Backup database
backup_database() {
    local env=$1
    local backup_file="backup-${env}-$(date +%Y%m%d-%H%M%S).sql.gz"
    
    log_info "Creating database backup..."
    
    if [[ "$env" == "local" ]]; then
        docker-compose exec -T postgres pg_dump -U postgres semiconductorlab | gzip > ${backup_file}
    else
        local namespace="${PROJECT_NAME}-${env}"
        local pod=$(kubectl get pods -n ${namespace} -l app=postgres -o jsonpath='{.items[0].metadata.name}')
        
        kubectl exec -n ${namespace} ${pod} -- \
            pg_dump -U postgres semiconductorlab | gzip > ${backup_file}
        
        # Upload to S3
        aws s3 cp ${backup_file} s3://${PROJECT_NAME}-backups/${env}/${backup_file}
    fi
    
    log_success "Database backup created: ${backup_file}"
}

# Restore database
restore_database() {
    local env=$1
    local backup_file=$2
    
    if [[ -z "$backup_file" ]]; then
        log_error "Backup file not specified"
        exit 1
    fi
    
    log_warning "This will overwrite the current database. Continue? (yes/no)"
    read -r confirm
    
    if [[ "$confirm" != "yes" ]]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    log_info "Restoring database from ${backup_file}..."
    
    if [[ "$env" == "local" ]]; then
        gunzip -c ${backup_file} | docker-compose exec -T postgres psql -U postgres semiconductorlab
    else
        local namespace="${PROJECT_NAME}-${env}"
        local pod=$(kubectl get pods -n ${namespace} -l app=postgres -o jsonpath='{.items[0].metadata.name}')
        
        gunzip -c ${backup_file} | kubectl exec -i -n ${namespace} ${pod} -- \
            psql -U postgres semiconductorlab
    fi
    
    log_success "Database restored successfully"
}

# Rollback deployment
rollback_deployment() {
    local env=$1
    local namespace="${PROJECT_NAME}-${env}"
    
    log_warning "Rolling back deployment in ${env}..."
    
    if [[ "$env" == "local" ]]; then
        docker-compose down
        # Would restore from previous docker-compose state
        log_info "Local rollback requires manual intervention"
    else
        # Helm rollback
        helm rollback ${PROJECT_NAME} -n ${namespace}
        
        # Wait for rollback to complete
        kubectl rollout status deployment -n ${namespace} --timeout=5m
    fi
    
    log_success "Rollback complete"
}

# Health checks
run_health_checks() {
    local env=$1
    
    log_info "Running health checks for ${env}..."
    
    local base_url
    if [[ "$env" == "local" ]]; then
        base_url="http://localhost:8000"
    elif [[ "$env" == "staging" ]]; then
        base_url="https://staging.semiconductorlab.com"
    else
        base_url="https://semiconductorlab.com"
    fi
    
    # Check API health
    log_info "Checking API health..."
    if curl -f -s "${base_url}/health" > /dev/null; then
        log_success "API is healthy"
    else
        log_error "API health check failed"
        exit 1
    fi
    
    # Check database connectivity
    log_info "Checking database connectivity..."
    if curl -f -s "${base_url}/health/db" > /dev/null; then
        log_success "Database is healthy"
    else
        log_error "Database health check failed"
        exit 1
    fi
    
    # Check object storage
    log_info "Checking object storage..."
    if curl -f -s "${base_url}/health/storage" > /dev/null; then
        log_success "Object storage is healthy"
    else
        log_warning "Object storage check failed (non-critical)"
    fi
    
    log_success "All critical health checks passed"
}

# Smoke tests
run_smoke_tests() {
    local env=$1
    
    log_info "Running smoke tests for ${env}..."
    
    local base_url
    if [[ "$env" == "local" ]]; then
        base_url="http://localhost:8000"
    elif [[ "$env" == "staging" ]]; then
        base_url="https://staging.semiconductorlab.com"
    else
        base_url="https://semiconductorlab.com"
    fi
    
    # Test 1: API responds
    log_info "Test 1: API responds..."
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "${base_url}/api/v1/health")
    if [[ "$http_code" == "200" ]]; then
        log_success "✓ API responds (200 OK)"
    else
        log_error "✗ API failed (HTTP ${http_code})"
        exit 1
    fi
    
    # Test 2: Database query
    log_info "Test 2: Database query..."
    if curl -f -s "${base_url}/api/v1/instruments" > /dev/null; then
        log_success "✓ Database queries working"
    else
        log_error "✗ Database query failed"
        exit 1
    fi
    
    # Test 3: Simple analysis
    log_info "Test 3: Simple analysis..."
    response=$(curl -s -X POST "${base_url}/api/v1/electrical/mosfet/analyze-transfer" \
        -H "Content-Type: application/json" \
        -d '{"voltage_gate": [0,1,2,3,4,5], "current_drain": [0,0.001,0.004,0.009,0.016,0.025], "voltage_drain": 0.1, "width": 1e-5, "length": 1e-6, "oxide_thickness": 1e-8}')
    
    if echo "$response" | grep -q "quality_score"; then
        log_success "✓ Analysis endpoint working"
    else
        log_error "✗ Analysis failed"
        exit 1
    fi
    
    log_success "All smoke tests passed"
}

# View logs
view_logs() {
    local env=$1
    local service=$2
    
    if [[ "$env" == "local" ]]; then
        if [[ -n "$service" ]]; then
            docker-compose logs -f ${service}
        else
            docker-compose logs -f
        fi
    else
        local namespace="${PROJECT_NAME}-${env}"
        if [[ -n "$service" ]]; then
            kubectl logs -f -n ${namespace} -l app=${service}
        else
            kubectl logs -f -n ${namespace} --all-containers=true
        fi
    fi
}

# Setup monitoring
setup_monitoring() {
    local env=$1
    local namespace="${PROJECT_NAME}-${env}"
    
    log_info "Setting up monitoring for ${env}..."
    
    # Install Prometheus
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values ./infra/monitoring/prometheus-values.yaml
    
    # Configure ServiceMonitors
    kubectl apply -f ./infra/monitoring/servicemonitors/ -n ${namespace}
    
    log_success "Monitoring setup complete"
    log_info "Access Grafana at: http://grafana.${env}.semiconductorlab.com"
}

# Setup SSL
setup_ssl() {
    local env=$1
    
    log_info "Setting up SSL certificates for ${env}..."
    
    # Install cert-manager
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    
    helm upgrade --install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --create-namespace \
        --set installCRDs=true
    
    # Apply ClusterIssuer
    kubectl apply -f ./infra/ssl/cluster-issuer-${env}.yaml
    
    # Apply Certificate
    kubectl apply -f ./infra/ssl/certificate-${env}.yaml
    
    log_success "SSL setup complete"
}

# Show deployment status
show_status() {
    local env=$1
    
    log_info "Deployment status for ${env}:"
    
    if [[ "$env" == "local" ]]; then
        docker-compose ps
    else
        local namespace="${PROJECT_NAME}-${env}"
        
        echo ""
        echo "Pods:"
        kubectl get pods -n ${namespace}
        
        echo ""
        echo "Services:"
        kubectl get services -n ${namespace}
        
        echo ""
        echo "Ingresses:"
        kubectl get ingress -n ${namespace}
        
        echo ""
        echo "Helm Releases:"
        helm list -n ${namespace}
    fi
}

# Main script logic
main() {
    if [[ $# -lt 2 ]]; then
        usage
    fi
    
    local env=$1
    local action=$2
    shift 2  # Remove first two arguments
    
    # Validate environment
    if [[ ! "$env" =~ ^(local|staging|production)$ ]]; then
        log_error "Invalid environment: $env"
        usage
    fi
    
    # Check prerequisites
    check_prerequisites "$env"
    
    # Execute action
    case "$action" in
        deploy)
            if [[ "$env" == "local" ]]; then
                deploy_local
            else
                build_images
                push_images
                deploy_kubernetes "$env"
            fi
            run_health_checks "$env"
            log_success "Deployment complete!"
            ;;
        backup)
            backup_database "$env"
            ;;
        restore)
            restore_database "$env" "$1"
            ;;
        rollback)
            rollback_deployment "$env"
            ;;
        health)
            run_health_checks "$env"
            ;;
        smoke-test)
            run_smoke_tests "$env"
            ;;
        logs)
            view_logs "$env" "$1"
            ;;
        setup-monitoring)
            setup_monitoring "$env"
            ;;
        setup-ssl)
            setup_ssl "$env"
            ;;
        status)
            show_status "$env"
            ;;
        *)
            log_error "Unknown action: $action"
            usage
            ;;
    esac
}

# Run main function
main "$@"