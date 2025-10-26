#!/bin/bash

################################################################################
# SESSION 14: ML/VM HUB - DEPLOYMENT SCRIPT
#
# Complete deployment automation for the Machine Learning and Virtual Metrology
# platform including feature engineering, model training, anomaly detection,
# and drift monitoring capabilities.
#
# Author: Semiconductor Lab Platform Team
# Date: October 2024
# Version: 1.0.0
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DEPLOY_ENV="${DEPLOY_ENV:-development}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-semiconductorlab}"
DB_USER="${DB_USER:-labuser}"
API_PORT="${API_PORT:-8014}"
PYTHON_VERSION="3.9"

# Logging
LOG_FILE="${PROJECT_ROOT}/deploy_session14_$(date +%Y%m%d_%H%M%S).log"

################################################################################
# UTILITY FUNCTIONS
################################################################################

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[✓]${NC} $*" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[!]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[✗]${NC} $*" | tee -a "$LOG_FILE"
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "$1 is not installed. Please install it first."
    fi
}

################################################################################
# PRE-FLIGHT CHECKS
################################################################################

preflight_checks() {
    log "Running pre-flight checks..."
    
    # Check required commands
    check_command python3
    check_command pip3
    check_command psql
    check_command docker
    check_command docker-compose
    
    # Check Python version
    PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$(printf '%s\n' "$PYTHON_VERSION" "$PYTHON_VER" | sort -V | head -n1)" != "$PYTHON_VERSION" ]; then
        error "Python $PYTHON_VERSION or higher required (found $PYTHON_VER)"
    fi
    
    # Check database connectivity
    if ! PGPASSWORD="${DB_PASSWORD}" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null; then
        warning "Cannot connect to database. Will attempt to create it."
    fi
    
    success "Pre-flight checks passed"
}

################################################################################
# PYTHON ENVIRONMENT SETUP
################################################################################

setup_python_environment() {
    log "Setting up Python environment..."
    
    # Create virtual environment
    if [ ! -d "${PROJECT_ROOT}/venv" ]; then
        python3 -m venv "${PROJECT_ROOT}/venv"
        success "Virtual environment created"
    else
        log "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "${PROJECT_ROOT}/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install core dependencies
    log "Installing core ML/VM dependencies..."
    pip install \
        numpy==1.24.3 \
        pandas==2.0.3 \
        scikit-learn==1.3.0 \
        scipy==1.11.1 \
        joblib==1.3.2 \
        pytest==7.4.0 \
        pytest-cov==4.1.0
    
    # Install ML frameworks
    log "Installing ML frameworks..."
    pip install \
        lightgbm==4.0.0 \
        prophet==1.1.4 \
        onnx==1.14.0 \
        onnxruntime==1.15.1 \
        skl2onnx==1.15.0
    
    # Install API dependencies
    log "Installing API dependencies..."
    pip install \
        fastapi==0.103.0 \
        uvicorn[standard]==0.23.2 \
        pydantic==2.3.0 \
        sqlalchemy==2.0.20 \
        psycopg2-binary==2.9.7 \
        python-multipart==0.0.6
    
    # Install development dependencies
    if [ "$DEPLOY_ENV" = "development" ]; then
        log "Installing development dependencies..."
        pip install \
            black==23.7.0 \
            flake8==6.1.0 \
            mypy==1.5.1 \
            ipython==8.14.0 \
            jupyter==1.0.0
    fi
    
    success "Python environment setup complete"
}

################################################################################
# DATABASE SETUP
################################################################################

setup_database() {
    log "Setting up database..."
    
    # Create database if it doesn't exist
    if ! PGPASSWORD="${DB_PASSWORD}" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log "Creating database $DB_NAME..."
        PGPASSWORD="${DB_PASSWORD}" createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
        success "Database created"
    else
        log "Database already exists"
    fi
    
    # Run migrations
    log "Running database migrations..."
    
    cat > /tmp/session14_migrations.sql << 'EOSQL'
-- Session 14: ML/VM Hub Tables

-- ML Models Registry
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'training',
    
    -- Model artifacts
    model_path VARCHAR(500),
    onnx_path VARCHAR(500),
    scaler_path VARCHAR(500),
    
    -- Feature information
    feature_names TEXT[],
    target_name VARCHAR(200),
    feature_importance JSONB,
    
    -- Training metadata
    training_data_size INTEGER,
    training_start TIMESTAMP,
    training_end TIMESTAMP,
    training_config JSONB,
    
    -- Performance metrics
    metrics JSONB,
    cv_scores JSONB,
    
    -- Deployment
    deployed_at TIMESTAMP,
    deployment_config JSONB,
    
    -- Monitoring
    last_prediction TIMESTAMP,
    prediction_count INTEGER DEFAULT 0,
    drift_detected BOOLEAN DEFAULT FALSE,
    last_drift_check TIMESTAMP,
    
    -- Metadata
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    
    UNIQUE(name, version)
);

CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status);
CREATE INDEX IF NOT EXISTS idx_ml_models_type ON ml_models(model_type);

-- Feature Store
CREATE TABLE IF NOT EXISTS feature_store (
    id SERIAL PRIMARY KEY,
    feature_set_name VARCHAR(200) NOT NULL,
    version VARCHAR(50) NOT NULL,
    
    -- Features
    feature_names TEXT[],
    feature_types JSONB,
    feature_definitions JSONB,
    
    -- Statistics
    feature_statistics JSONB,
    correlation_matrix JSONB,
    
    -- Lineage
    source_tables TEXT[],
    transformation_pipeline JSONB,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    UNIQUE(feature_set_name, version)
);

-- Model Predictions
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    
    -- Input/Output
    features JSONB,
    prediction DOUBLE PRECISION,
    prediction_proba DOUBLE PRECISION[],
    
    -- Confidence
    confidence_score DOUBLE PRECISION,
    uncertainty DOUBLE PRECISION,
    
    -- Context
    sample_id VARCHAR(100),
    run_id INTEGER,
    instrument_id INTEGER,
    
    -- Actual value (if available)
    actual_value DOUBLE PRECISION,
    prediction_error DOUBLE PRECISION,
    
    -- Metadata
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    inference_time_ms DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON model_predictions(timestamp);

-- Drift Reports
CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    
    -- Drift detection
    drift_type VARCHAR(50),
    drift_detected BOOLEAN,
    drift_score DOUBLE PRECISION,
    drift_threshold DOUBLE PRECISION,
    
    -- Analysis window
    reference_start TIMESTAMP,
    reference_end TIMESTAMP,
    current_start TIMESTAMP,
    current_end TIMESTAMP,
    
    -- Details
    affected_features TEXT[],
    feature_drift_scores JSONB,
    statistical_tests JSONB,
    
    -- Recommendations
    recommended_action VARCHAR(50),
    retrain_recommended BOOLEAN,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_drift_model ON drift_reports(model_id);
CREATE INDEX IF NOT EXISTS idx_drift_detected ON drift_reports(drift_detected);

-- Anomaly Detections
CREATE TABLE IF NOT EXISTS anomaly_detections (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    
    -- Anomaly info
    is_anomaly BOOLEAN,
    anomaly_score DOUBLE PRECISION,
    anomaly_type VARCHAR(50),
    
    -- Context
    sample_id VARCHAR(100),
    run_id INTEGER,
    timestamp TIMESTAMP,
    
    -- Features at time of anomaly
    features JSONB,
    feature_contributions JSONB,
    
    -- Explanation
    likely_causes TEXT[],
    similar_anomalies JSONB,
    
    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    
    -- Metadata
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_anomaly_model ON anomaly_detections(model_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_unresolved ON anomaly_detections(resolved) WHERE NOT resolved;

-- Maintenance Predictions
CREATE TABLE IF NOT EXISTS maintenance_predictions (
    id SERIAL PRIMARY KEY,
    instrument_id INTEGER NOT NULL,
    model_id INTEGER REFERENCES ml_models(id),
    
    -- Prediction
    failure_probability DOUBLE PRECISION,
    estimated_rul_hours DOUBLE PRECISION,  -- Remaining useful life
    confidence_interval DOUBLE PRECISION[],
    
    -- Risk assessment
    risk_level VARCHAR(20),
    recommended_action VARCHAR(50),
    urgency_score DOUBLE PRECISION,
    
    -- Features
    health_indicators JSONB,
    degradation_rate DOUBLE PRECISION,
    
    -- Recommendations
    maintenance_type VARCHAR(50),
    estimated_cost DOUBLE PRECISION,
    suggested_date TIMESTAMP,
    
    -- Outcome tracking
    actual_failure_date TIMESTAMP,
    maintenance_performed BOOLEAN DEFAULT FALSE,
    maintenance_date TIMESTAMP,
    
    -- Metadata
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_maintenance_instrument ON maintenance_predictions(instrument_id);
CREATE INDEX IF NOT EXISTS idx_maintenance_risk ON maintenance_predictions(risk_level);

-- Create views for monitoring
CREATE OR REPLACE VIEW v_active_models AS
SELECT 
    m.*,
    COUNT(DISTINCT p.id) as total_predictions,
    MAX(p.timestamp) as last_used,
    AVG(CASE WHEN p.actual_value IS NOT NULL 
        THEN ABS(p.prediction - p.actual_value) / NULLIF(p.actual_value, 0)
        ELSE NULL END) * 100 as avg_error_percent
FROM ml_models m
LEFT JOIN model_predictions p ON m.id = p.model_id
WHERE m.status IN ('ready', 'deployed')
GROUP BY m.id;

CREATE OR REPLACE VIEW v_recent_anomalies AS
SELECT 
    ad.*,
    m.name as model_name,
    m.version as model_version
FROM anomaly_detections ad
JOIN ml_models m ON ad.model_id = m.id
WHERE ad.detected_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY ad.detected_at DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO labuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO labuser;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;

EOSQL
    
    PGPASSWORD="${DB_PASSWORD}" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f /tmp/session14_migrations.sql
    
    success "Database setup complete"
}

################################################################################
# APPLICATION DEPLOYMENT
################################################################################

deploy_application() {
    log "Deploying ML/VM Hub application..."
    
    # Copy implementation files
    mkdir -p "${PROJECT_ROOT}/ml_vm_hub"
    cp session14_vm_ml_complete_implementation.py "${PROJECT_ROOT}/ml_vm_hub/"
    
    # Create configuration file
    cat > "${PROJECT_ROOT}/ml_vm_hub/config.py" << EOPYCONFIG
"""ML/VM Hub Configuration"""
import os

class Config:
    # Database
    DB_HOST = os.getenv('DB_HOST', '${DB_HOST}')
    DB_PORT = os.getenv('DB_PORT', '${DB_PORT}')
    DB_NAME = os.getenv('DB_NAME', '${DB_NAME}')
    DB_USER = os.getenv('DB_USER', '${DB_USER}')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # API
    API_HOST = '0.0.0.0'
    API_PORT = ${API_PORT}
    
    # ML
    MODEL_STORE_PATH = os.getenv('MODEL_STORE_PATH', './models')
    FEATURE_STORE_PATH = os.getenv('FEATURE_STORE_PATH', './features')
    
    # Monitoring
    DRIFT_CHECK_INTERVAL_HOURS = 24
    ANOMALY_THRESHOLD = 0.5
    
config = Config()
EOPYCONFIG
    
    # Create model storage directories
    mkdir -p "${PROJECT_ROOT}/ml_vm_hub/models"
    mkdir -p "${PROJECT_ROOT}/ml_vm_hub/features"
    mkdir -p "${PROJECT_ROOT}/ml_vm_hub/logs"
    
    success "Application deployed"
}

################################################################################
# DOCKER SETUP
################################################################################

setup_docker() {
    log "Setting up Docker services..."
    
    # Create Docker Compose file
    cat > "${PROJECT_ROOT}/docker-compose-session14.yml" << 'EODOCKER'
version: '3.8'

services:
  ml-vm-hub:
    build:
      context: .
      dockerfile: Dockerfile.session14
    container_name: ml-vm-hub
    ports:
      - "8014:8014"
    environment:
      - DB_HOST=${DB_HOST:-db}
      - DB_PORT=${DB_PORT:-5432}
      - DB_NAME=${DB_NAME:-semiconductorlab}
      - DB_USER=${DB_USER:-labuser}
      - DB_PASSWORD=${DB_PASSWORD}
      - PYTHONUNBUFFERED=1
    volumes:
      - ./ml_vm_hub:/app
      - ./models:/app/models
      - ./features:/app/features
    depends_on:
      - db
    restart: unless-stopped
    networks:
      - labnet

  ml-worker:
    build:
      context: .
      dockerfile: Dockerfile.session14
    container_name: ml-worker
    command: python -m celery -A ml_vm_hub.tasks worker --loglevel=info
    environment:
      - DB_HOST=${DB_HOST:-db}
      - DB_PORT=${DB_PORT:-5432}
      - DB_NAME=${DB_NAME:-semiconductorlab}
      - DB_USER=${DB_USER:-labuser}
      - DB_PASSWORD=${DB_PASSWORD}
    volumes:
      - ./ml_vm_hub:/app
      - ./models:/app/models
    depends_on:
      - db
      - redis
    restart: unless-stopped
    networks:
      - labnet

  redis:
    image: redis:7-alpine
    container_name: ml-redis
    ports:
      - "6379:6379"
    restart: unless-stopped
    networks:
      - labnet

networks:
  labnet:
    external: true

EODOCKER
    
    # Create Dockerfile
    cat > "${PROJECT_ROOT}/Dockerfile.session14" << 'EODOCKERFILE'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-session14.txt .
RUN pip install --no-cache-dir -r requirements-session14.txt

# Copy application
COPY ml_vm_hub/ ./ml_vm_hub/

# Create directories
RUN mkdir -p models features logs

EXPOSE 8014

CMD ["uvicorn", "ml_vm_hub.session14_vm_ml_complete_implementation:app", "--host", "0.0.0.0", "--port", "8014"]
EODOCKERFILE
    
    # Create requirements file
    cat > "${PROJECT_ROOT}/requirements-session14.txt" << 'EOREQ'
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.11.1
lightgbm==4.0.0
prophet==1.1.4
onnx==1.14.0
onnxruntime==1.15.1
skl2onnx==1.15.0
fastapi==0.103.0
uvicorn[standard]==0.23.2
pydantic==2.3.0
sqlalchemy==2.0.20
psycopg2-binary==2.9.7
python-multipart==0.0.6
joblib==1.3.2
celery==5.3.1
redis==4.6.0
EOREQ
    
    success "Docker setup complete"
}

################################################################################
# TESTS
################################################################################

run_tests() {
    log "Running tests..."
    
    source "${PROJECT_ROOT}/venv/bin/activate"
    
    # Run unit tests
    pytest test_session14_integration.py -v --cov=session14_vm_ml_complete_implementation --cov-report=html --cov-report=term
    
    # Check coverage
    COVERAGE=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
    if [ "${COVERAGE%.*}" -lt 80 ]; then
        warning "Test coverage is below 80% ($COVERAGE%)"
    else
        success "Test coverage: $COVERAGE%"
    fi
}

################################################################################
# SERVICE START
################################################################################

start_services() {
    log "Starting services..."
    
    if [ "$DEPLOY_ENV" = "production" ]; then
        log "Starting with Docker Compose..."
        docker-compose -f docker-compose-session14.yml up -d
        success "Services started with Docker"
    else
        log "Starting development server..."
        source "${PROJECT_ROOT}/venv/bin/activate"
        cd "${PROJECT_ROOT}/ml_vm_hub"
        
        # Start FastAPI in background
        uvicorn session14_vm_ml_complete_implementation:app \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --reload \
            > "${PROJECT_ROOT}/ml_vm_hub/logs/api.log" 2>&1 &
        
        echo $! > "${PROJECT_ROOT}/ml_vm_hub/api.pid"
        success "Development server started on port $API_PORT"
    fi
}

################################################################################
# HEALTH CHECK
################################################################################

health_check() {
    log "Performing health check..."
    
    # Wait for service to start
    sleep 5
    
    # Check API
    MAX_RETRIES=10
    RETRY=0
    while [ $RETRY -lt $MAX_RETRIES ]; do
        if curl -s "http://localhost:$API_PORT/health" > /dev/null; then
            success "API health check passed"
            break
        fi
        RETRY=$((RETRY + 1))
        log "Waiting for API to be ready... ($RETRY/$MAX_RETRIES)"
        sleep 2
    done
    
    if [ $RETRY -eq $MAX_RETRIES ]; then
        error "API health check failed"
    fi
    
    # Check database connectivity
    if python3 << EOCHECK
import psycopg2
try:
    conn = psycopg2.connect(
        host='${DB_HOST}',
        port=${DB_PORT},
        database='${DB_NAME}',
        user='${DB_USER}',
        password='${DB_PASSWORD}'
    )
    conn.close()
    print("OK")
except Exception as e:
    print(f"Error: {e}")
    exit(1)
EOCHECK
    then
        success "Database connectivity check passed"
    else
        error "Database connectivity check failed"
    fi
}

################################################################################
# MONITORING SETUP
################################################################################

setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create monitoring script
    cat > "${PROJECT_ROOT}/ml_vm_hub/monitor.py" << 'EOMONITOR'
#!/usr/bin/env python3
"""ML/VM Hub Monitoring"""
import time
import psycopg2
from datetime import datetime, timedelta

def check_model_performance():
    """Check model performance metrics"""
    # Implementation here
    pass

def check_drift():
    """Check for model drift"""
    # Implementation here
    pass

def check_anomalies():
    """Check for unresolved anomalies"""
    # Implementation here
    pass

if __name__ == '__main__':
    while True:
        check_model_performance()
        check_drift()
        check_anomalies()
        time.sleep(3600)  # Check every hour
EOMONITOR
    
    chmod +x "${PROJECT_ROOT}/ml_vm_hub/monitor.py"
    
    success "Monitoring setup complete"
}

################################################################################
# MAIN DEPLOYMENT FLOW
################################################################################

main() {
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║       SESSION 14: ML/VM HUB DEPLOYMENT                        ║"
    echo "║       Machine Learning & Virtual Metrology Platform           ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    
    log "Starting deployment..."
    log "Environment: $DEPLOY_ENV"
    log "Database: $DB_HOST:$DB_PORT/$DB_NAME"
    log "API Port: $API_PORT"
    echo ""
    
    preflight_checks
    setup_python_environment
    setup_database
    deploy_application
    
    if [ "$DEPLOY_ENV" = "production" ]; then
        setup_docker
    fi
    
    run_tests
    start_services
    health_check
    setup_monitoring
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                   DEPLOYMENT COMPLETE!                        ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "API Endpoint: http://localhost:$API_PORT"
    echo "API Docs: http://localhost:$API_PORT/docs"
    echo "Health Check: http://localhost:$API_PORT/health"
    echo ""
    echo "Log file: $LOG_FILE"
    echo ""
    
    success "Session 14 deployment successful!"
}

# Run main deployment
main "$@"
