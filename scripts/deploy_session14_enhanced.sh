#!/bin/bash

###############################################################################
# SESSION 14 ENHANCED DEPLOYMENT SCRIPT
# Complete ML/VM Suite with Advanced Features
#
# This script deploys the enhanced Session 14 with:
# - AutoML capabilities
# - Model explainability
# - A/B testing framework
# - Advanced drift detection
# - Production monitoring
# - Model governance
#
# Author: Semiconductor Lab Platform Team
# Date: October 2024
# Version: 2.0.0
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
DEPLOY_ENV=${DEPLOY_ENV:-development}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${BASE_DIR}/venv"

# Enhanced dependencies
PYTHON_VERSION="3.9"
ENHANCED_PACKAGES=(
    "optuna>=3.0.0"
    "shap>=0.42.0"
    "lime>=0.2.0"
    "prometheus-client>=0.17.0"
    "xgboost>=1.7.0"
    "catboost>=1.2.0"
    "statsmodels>=0.14.0"
)

###############################################################################
# Utility Functions
###############################################################################

print_header() {
    echo -e "${BLUE}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "$1"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed"
        return 1
    fi
    print_success "$1 is installed"
    return 0
}

###############################################################################
# Pre-flight Checks
###############################################################################

preflight_checks() {
    print_header "Running Pre-flight Checks"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version | awk '{print $2}')
        print_success "Python version: $PYTHON_VER"
    else
        print_error "Python 3 not found"
        exit 1
    fi
    
    # Check required commands
    check_command "pip3" || exit 1
    check_command "psql" || print_warning "PostgreSQL client not found (optional)"
    check_command "docker" || print_warning "Docker not found (optional)"
    check_command "git" || exit 1
    
    # Check disk space
    AVAILABLE_SPACE=$(df -h "$BASE_DIR" | tail -1 | awk '{print $4}')
    print_success "Available disk space: $AVAILABLE_SPACE"
    
    # Check memory
    if command -v free &> /dev/null; then
        AVAILABLE_MEM=$(free -h | grep Mem | awk '{print $7}')
        print_success "Available memory: $AVAILABLE_MEM"
    fi
    
    echo ""
}

###############################################################################
# Virtual Environment Setup
###############################################################################

setup_venv() {
    print_header "Setting Up Python Virtual Environment"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists"
        read -p "Recreate virtual environment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            print_success "Using existing virtual environment"
            return 0
        fi
    fi
    
    # Create venv
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
    
    # Activate
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    print_success "pip upgraded"
    
    echo ""
}

###############################################################################
# Install Dependencies
###############################################################################

install_dependencies() {
    print_header "Installing Python Dependencies"
    
    source "$VENV_DIR/bin/activate"
    
    # Install base requirements
    if [ -f "$BASE_DIR/requirements.txt" ]; then
        print_success "Installing from requirements.txt..."
        pip install -r "$BASE_DIR/requirements.txt"
    fi
    
    # Install enhanced packages
    print_success "Installing enhanced ML packages..."
    for package in "${ENHANCED_PACKAGES[@]}"; do
        echo "  Installing $package..."
        pip install "$package" || print_warning "Failed to install $package"
    done
    
    # Install monitoring tools
    print_success "Installing monitoring tools..."
    pip install prometheus-client grafana-api
    
    # Install optional packages
    print_success "Installing optional packages..."
    pip install boruta py-spy memory-profiler || print_warning "Some optional packages failed"
    
    # List installed packages
    echo ""
    print_success "Installed packages:"
    pip list | grep -E "optuna|shap|lime|prometheus|xgboost|catboost|lightgbm"
    
    echo ""
}

###############################################################################
# Database Setup - Enhanced Schema
###############################################################################

setup_database() {
    print_header "Setting Up Enhanced Database Schema"
    
    # Database configuration
    DB_HOST=${DB_HOST:-localhost}
    DB_PORT=${DB_PORT:-5432}
    DB_NAME=${DB_NAME:-semiconductorlab}
    DB_USER=${DB_USER:-labuser}
    
    # Check PostgreSQL connection
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1" &> /dev/null; then
        print_warning "Cannot connect to PostgreSQL"
        print_warning "Please ensure PostgreSQL is running and credentials are correct"
        read -p "Continue without database setup? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        return 0
    fi
    
    print_success "Connected to PostgreSQL"
    
    # Create enhanced tables
    cat > /tmp/session14_enhanced_schema.sql << 'EOF'
-- Enhanced ML Models table
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT gen_random_uuid() UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'training',
    
    -- Model artifacts
    model_path VARCHAR(500),
    onnx_path VARCHAR(500),
    model_hash VARCHAR(64),
    
    -- Configuration
    config JSONB,
    hyperparameters JSONB,
    
    -- Training metadata
    training_data_hash VARCHAR(64),
    feature_names TEXT[],
    target_name VARCHAR(255),
    n_features INTEGER,
    n_samples_train INTEGER,
    n_samples_test INTEGER,
    
    -- Performance metrics
    metrics_train JSONB,
    metrics_test JSONB,
    metrics_cv JSONB,
    feature_importance JSONB,
    
    -- Governance
    created_by VARCHAR(255),
    approved_by VARCHAR(255),
    approval_date TIMESTAMP,
    
    -- Deployment
    deployment_id VARCHAR(100),
    deployed_at TIMESTAMP,
    endpoint_url VARCHAR(500),
    
    -- Monitoring
    prediction_count INTEGER DEFAULT 0,
    last_prediction_at TIMESTAMP,
    current_drift_score FLOAT,
    last_drift_check_at TIMESTAMP,
    
    -- A/B Testing
    ab_test_id VARCHAR(100),
    ab_test_variant VARCHAR(50),
    ab_test_traffic_percentage FLOAT,
    
    -- Lineage
    parent_model_id INTEGER REFERENCES ml_models(id),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ml_models_name_version ON ml_models(name, version);
CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status);
CREATE INDEX IF NOT EXISTS idx_ml_models_algorithm ON ml_models(algorithm);
CREATE INDEX IF NOT EXISTS idx_ml_models_created_at ON ml_models(created_at);

-- Model Explanations table
CREATE TABLE IF NOT EXISTS model_explanations (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id) NOT NULL,
    explanation_method VARCHAR(50) NOT NULL,
    
    -- Global explanations
    global_feature_importance JSONB,
    global_shap_values JSONB,
    
    -- Interactions
    interaction_values JSONB,
    
    -- Partial dependence
    partial_dependence_plots JSONB,
    
    -- Summary
    summary_plot_data JSONB,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Prediction Logs table (enhanced)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id) NOT NULL,
    prediction_id UUID DEFAULT gen_random_uuid() UNIQUE,
    
    -- Input/Output
    features JSONB NOT NULL,
    prediction FLOAT,
    prediction_proba FLOAT[],
    uncertainty FLOAT,
    
    -- Explanation
    shap_values JSONB,
    lime_explanation JSONB,
    
    -- Context
    run_id INTEGER,
    sample_id INTEGER,
    wafer_id INTEGER,
    device_id INTEGER,
    
    -- Quality flags
    is_anomalous BOOLEAN DEFAULT FALSE,
    confidence_score FLOAT,
    drift_detected BOOLEAN DEFAULT FALSE,
    
    -- Performance
    latency_ms FLOAT,
    
    -- A/B Testing
    ab_test_variant VARCHAR(50),
    
    -- Feedback loop
    actual_value FLOAT,
    feedback_received BOOLEAN DEFAULT FALSE,
    feedback_timestamp TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_created ON prediction_logs(model_id, created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_run ON prediction_logs(run_id);

-- A/B Tests table
CREATE TABLE IF NOT EXISTS ab_tests (
    id SERIAL PRIMARY KEY,
    test_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Models
    control_model_id INTEGER REFERENCES ml_models(id),
    variant_models JSONB,
    
    -- Traffic allocation
    traffic_allocation JSONB,
    
    -- Configuration
    min_sample_size INTEGER DEFAULT 100,
    max_duration_days INTEGER DEFAULT 14,
    success_metric VARCHAR(100),
    
    -- Status
    status VARCHAR(50),
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    
    -- Results
    results JSONB,
    winner_variant VARCHAR(50),
    statistical_significance FLOAT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- AutoML Experiments table
CREATE TABLE IF NOT EXISTS automl_experiments (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    
    -- Configuration
    target_metric VARCHAR(100),
    n_trials INTEGER,
    timeout_seconds INTEGER,
    
    -- Search space
    algorithms TEXT[],
    search_space JSONB,
    
    -- Best trial
    best_trial_id VARCHAR(100),
    best_model_id INTEGER REFERENCES ml_models(id),
    best_score FLOAT,
    best_hyperparameters JSONB,
    
    -- All trials
    all_trials JSONB,
    
    -- Status
    status VARCHAR(50),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Model Audit Log
CREATE TABLE IF NOT EXISTS model_audit_log (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    user_id VARCHAR(255),
    ip_address INET,
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_model ON model_audit_log(model_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON model_audit_log(created_at);

-- Views
CREATE OR REPLACE VIEW v_active_models AS
SELECT 
    m.id,
    m.name,
    m.version,
    m.algorithm,
    m.status,
    m.metrics_test->>'r2' as test_r2,
    m.metrics_test->>'rmse' as test_rmse,
    m.prediction_count,
    m.current_drift_score,
    m.deployed_at,
    m.created_at
FROM ml_models m
WHERE m.status IN ('ready', 'deployed', 'ab_testing')
ORDER BY m.created_at DESC;

CREATE OR REPLACE VIEW v_model_performance_summary AS
SELECT 
    m.name,
    m.algorithm,
    COUNT(*) as version_count,
    AVG((m.metrics_test->>'r2')::float) as avg_r2,
    MAX((m.metrics_test->>'r2')::float) as best_r2,
    SUM(m.prediction_count) as total_predictions
FROM ml_models m
WHERE m.metrics_test IS NOT NULL
GROUP BY m.name, m.algorithm
ORDER BY avg_r2 DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO labuser;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO labuser;

EOF
    
    # Execute SQL
    print_success "Creating enhanced database schema..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f /tmp/session14_enhanced_schema.sql
    
    if [ $? -eq 0 ]; then
        print_success "Database schema created successfully"
    else
        print_error "Failed to create database schema"
        exit 1
    fi
    
    # Cleanup
    rm /tmp/session14_enhanced_schema.sql
    
    echo ""
}

###############################################################################
# Create Test Data
###############################################################################

create_test_data() {
    print_header "Creating Enhanced Test Data"
    
    source "$VENV_DIR/bin/activate"
    
    python3 << 'PYTHON_EOF'
import numpy as np
import pandas as pd
from pathlib import Path
import json

np.random.seed(42)

# Test data directory
test_data_dir = Path("test_data/session14_enhanced")
test_data_dir.mkdir(parents=True, exist_ok=True)

print("Generating enhanced test datasets...")

# 1. Virtual Metrology Training Data
print("  1. VM training data (5000 samples)...")
n_samples = 5000
vm_data = pd.DataFrame({
    'temperature': np.random.normal(350, 20, n_samples),
    'pressure': np.random.normal(100, 10, n_samples),
    'flow_rate': np.random.normal(50, 5, n_samples),
    'power': np.random.normal(1000, 50, n_samples),
    'time': np.random.normal(60, 5, n_samples),
    'chamber_age': np.random.uniform(0, 1000, n_samples),
    'maintenance_cycles': np.random.poisson(5, n_samples),
    'previous_run_thickness': np.random.normal(100, 10, n_samples),
})

# Target: thickness (with realistic dependencies)
vm_data['thickness'] = (
    0.3 * vm_data['temperature'] +
    0.2 * vm_data['power'] / 10 +
    0.5 * vm_data['time'] +
    np.random.normal(0, 3, n_samples)
)

vm_data.to_csv(test_data_dir / 'vm_training_data.csv', index=False)
print(f"    Saved: {test_data_dir / 'vm_training_data.csv'}")

# 2. AutoML Comparison Data
print("  2. AutoML comparison data...")
automl_results = {
    "manual_tuning": {"r2": 0.8932, "time_hours": 4.0},
    "grid_search": {"r2": 0.8945, "time_hours": 8.0},
    "random_search": {"r2": 0.8821, "time_hours": 2.0},
    "automl_optuna": {"r2": 0.9187, "time_hours": 0.75},
    "automl_ensemble": {"r2": 0.9342, "time_hours": 1.5}
}
with open(test_data_dir / 'automl_comparison.json', 'w') as f:
    json.dump(automl_results, f, indent=2)
print(f"    Saved: {test_data_dir / 'automl_comparison.json'}")

# 3. Drift Detection Data
print("  3. Drift detection data...")
# Reference distribution
ref_data = pd.DataFrame({
    'feature_1': np.random.normal(100, 10, 1000),
    'feature_2': np.random.normal(50, 5, 1000),
    'feature_3': np.random.exponential(2, 1000),
})

# Current distribution (with drift)
curr_data = pd.DataFrame({
    'feature_1': np.random.normal(105, 12, 1000),  # Mean shifted
    'feature_2': np.random.normal(50, 8, 1000),    # Variance increased
    'feature_3': np.random.exponential(2.5, 1000), # Distribution changed
})

ref_data.to_csv(test_data_dir / 'drift_reference.csv', index=False)
curr_data.to_csv(test_data_dir / 'drift_current.csv', index=False)
print(f"    Saved: drift_reference.csv, drift_current.csv")

# 4. Time Series Data for Decomposition
print("  4. Time series decomposition data...")
dates = pd.date_range('2024-01-01', periods=365, freq='D')
trend = np.linspace(100, 150, 365)
seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly seasonality
noise = np.random.normal(0, 2, 365)
ts_data = pd.DataFrame({
    'date': dates,
    'value': trend + seasonal + noise
})
ts_data.to_csv(test_data_dir / 'timeseries_data.csv', index=False)
print(f"    Saved: {test_data_dir / 'timeseries_data.csv'}")

# 5. A/B Testing Data
print("  5. A/B testing simulation data...")
ab_data = []
for variant in ['control', 'variant_a', 'variant_b']:
    base_error = {'control': 3.2, 'variant_a': 2.8, 'variant_b': 2.9}[variant]
    errors = np.random.normal(base_error, 0.5, 150)
    for error in errors:
        ab_data.append({
            'variant': variant,
            'prediction_error': error,
            'timestamp': pd.Timestamp.now().isoformat()
        })

pd.DataFrame(ab_data).to_csv(test_data_dir / 'ab_test_data.csv', index=False)
print(f"    Saved: {test_data_dir / 'ab_test_data.csv'}")

# 6. Feature Importance Data
print("  6. Feature importance reference data...")
feature_importance = {
    'temperature': 0.324,
    'power': 0.218,
    'time': 0.187,
    'pressure': 0.143,
    'flow_rate': 0.098,
    'chamber_age': 0.030
}
with open(test_data_dir / 'feature_importance.json', 'w') as f:
    json.dump(feature_importance, f, indent=2)
print(f"    Saved: {test_data_dir / 'feature_importance.json'}")

# 7. Causal Analysis Data
print("  7. Causal analysis data...")
n_causal = 2000
causal_data = pd.DataFrame({
    'treatment': np.random.binomial(1, 0.5, n_causal),
    'confounder_1': np.random.normal(0, 1, n_causal),
    'confounder_2': np.random.normal(0, 1, n_causal),
})
# Outcome depends on treatment and confounders
causal_data['outcome'] = (
    3.5 * causal_data['treatment'] +  # Treatment effect
    2.0 * causal_data['confounder_1'] +
    1.5 * causal_data['confounder_2'] +
    np.random.normal(0, 1, n_causal)
)
causal_data.to_csv(test_data_dir / 'causal_data.csv', index=False)
print(f"    Saved: {test_data_dir / 'causal_data.csv'}")

print("\n✓ All enhanced test data created successfully!")
print(f"  Location: {test_data_dir.absolute()}")

PYTHON_EOF
    
    print_success "Enhanced test data created"
    echo ""
}

###############################################################################
# Configure Monitoring
###############################################################################

setup_monitoring() {
    print_header "Setting Up Production Monitoring"
    
    # Create Prometheus configuration
    mkdir -p "${BASE_DIR}/monitoring/prometheus"
    
    cat > "${BASE_DIR}/monitoring/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml_platform'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'ml_vm_suite'
          session: 'session_14_enhanced'

  - job_name: 'fastapi_metrics'
    static_configs:
      - targets: ['localhost:8014']
        labels:
          service: 'ml_api'
          
rule_files:
  - 'ml_alerts.yml'
EOF
    
    # Create alert rules
    cat > "${BASE_DIR}/monitoring/prometheus/ml_alerts.yml" << 'EOF'
groups:
  - name: ml_platform_alerts
    interval: 30s
    rules:
      # Model Performance Degradation
      - alert: ModelPerformanceDegraded
        expr: ml_model_r2_score < 0.85
        for: 5m
        labels:
          severity: warning
          component: ml_model
        annotations:
          summary: "Model {{ $labels.model_name }} R² dropped below 0.85"
          description: "Current R²: {{ $value }}"

      # High Drift Score
      - alert: HighDriftScore
        expr: ml_drift_score > 0.3
        for: 10m
        labels:
          severity: warning
          component: drift_detection
        annotations:
          summary: "High drift detected for {{ $labels.model_name }}"
          description: "Drift score: {{ $value }}"

      # Critical Drift
      - alert: CriticalDrift
        expr: ml_drift_score > 0.5
        for: 5m
        labels:
          severity: critical
          component: drift_detection
        annotations:
          summary: "CRITICAL: Severe drift for {{ $labels.model_name }}"
          description: "Immediate retraining required. Drift score: {{ $value }}"

      # High Anomaly Rate
      - alert: HighAnomalyRate
        expr: rate(ml_anomalies_detected_total[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
          component: anomaly_detection
        annotations:
          summary: "High anomaly rate detected"
          description: "Anomaly rate: {{ $value }} per second"

      # Slow Predictions
      - alert: SlowPredictions
        expr: histogram_quantile(0.95, ml_prediction_latency_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
          component: inference
        annotations:
          summary: "Slow predictions for {{ $labels.model_name }}"
          description: "P95 latency: {{ $value }}s"
EOF
    
    print_success "Prometheus configuration created"
    
    # Create Grafana dashboard
    mkdir -p "${BASE_DIR}/monitoring/grafana"
    
    cat > "${BASE_DIR}/monitoring/grafana/ml_dashboard.json" << 'EOF'
{
  "dashboard": {
    "title": "ML/VM Platform - Enhanced Monitoring",
    "tags": ["ml", "vm", "session14"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Model Performance (R²)",
        "type": "graph",
        "targets": [{
          "expr": "ml_model_r2_score"
        }]
      },
      {
        "title": "Prediction Latency (p95)",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, ml_prediction_latency_seconds_bucket)"
        }]
      },
      {
        "title": "Drift Scores",
        "type": "graph",
        "targets": [{
          "expr": "ml_drift_score"
        }]
      },
      {
        "title": "Anomaly Detection Rate",
        "type": "graph",
        "targets": [{
          "expr": "rate(ml_anomalies_detected_total[5m])"
        }]
      },
      {
        "title": "Model Training Jobs",
        "type": "stat",
        "targets": [{
          "expr": "ml_model_training_total"
        }]
      },
      {
        "title": "Total Predictions",
        "type": "stat",
        "targets": [{
          "expr": "ml_predictions_total"
        }]
      }
    ]
  }
}
EOF
    
    print_success "Grafana dashboard template created"
    
    echo ""
    print_warning "To start monitoring:"
    echo "  1. Start Prometheus: docker run -p 9090:9090 -v ./monitoring/prometheus:/etc/prometheus prom/prometheus"
    echo "  2. Start Grafana: docker run -p 3000:3000 grafana/grafana"
    echo "  3. Import dashboard from monitoring/grafana/ml_dashboard.json"
    echo ""
}

###############################################################################
# Docker Setup
###############################################################################

setup_docker() {
    print_header "Setting Up Docker Configuration"
    
    # Create Dockerfile
    cat > "${BASE_DIR}/Dockerfile.enhanced" << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements_enhanced.txt ./

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_enhanced.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8014 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8014/health')"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8014"]
EOF
    
    # Create docker-compose.yml
    cat > "${BASE_DIR}/docker-compose.enhanced.yml" << 'EOF'
version: '3.8'

services:
  ml-platform:
    build:
      context: .
      dockerfile: Dockerfile.enhanced
    ports:
      - "8014:8014"
      - "8000:8000"
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=semiconductorlab
      - DB_USER=labuser
      - DB_PASSWORD=labpassword
      - MODEL_STORE_PATH=/data/models
      - ENABLE_PROMETHEUS=true
    volumes:
      - ./data:/data
      - ./models:/models
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=semiconductorlab
      - POSTGRES_USER=labuser
      - POSTGRES_PASSWORD=labpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: ml_platform_network
EOF
    
    print_success "Docker configuration created"
    
    echo ""
    print_warning "To deploy with Docker:"
    echo "  docker-compose -f docker-compose.enhanced.yml up -d"
    echo ""
}

###############################################################################
# Health Checks
###############################################################################

health_checks() {
    print_header "Running Health Checks"
    
    source "$VENV_DIR/bin/activate"
    
    # Test imports
    print_success "Testing Python imports..."
    python3 -c "
import sys
modules = [
    'numpy', 'pandas', 'sklearn', 'scipy',
    'optuna', 'shap', 'lime',
    'prometheus_client',
    'lightgbm', 'xgboost', 'catboost'
]
failed = []
for module in modules:
    try:
        __import__(module)
        print(f'  ✓ {module}')
    except ImportError:
        print(f'  ✗ {module} (optional)')
        if module in ['optuna', 'shap']:
            failed.append(module)

if failed:
    print(f'\n⚠️  Critical modules missing: {failed}')
    sys.exit(1)
" || exit 1
    
    print_success "All critical imports successful"
    
    # Check files
    print_success "Checking enhanced implementation files..."
    files_to_check=(
        "session14_enhanced_implementation.py"
        "session14_enhanced_part2.py"
        "SESSION_14_ENHANCED_README.md"
    )
    
    for file in "${files_to_check[@]}"; do
        if [ -f "$file" ]; then
            print_success "  $file exists"
        else
            print_error "  $file missing"
        fi
    done
    
    echo ""
}

###############################################################################
# Generate Documentation
###############################################################################

generate_docs() {
    print_header "Generating Enhanced Documentation"
    
    mkdir -p "${BASE_DIR}/docs/enhanced"
    
    # Create quick start guide
    cat > "${BASE_DIR}/docs/enhanced/QUICK_START.md" << 'EOF'
# Session 14 Enhanced - Quick Start Guide

## Installation

./deploy_session14_enhanced.sh

## First Steps

### 1. Run AutoML

from session14_enhanced_implementation import AutoMLEngine, AutoMLConfig

config = AutoMLConfig(target_metric="r2", n_trials=50)
engine = AutoMLEngine(config)
model, results = engine.optimize(X_train, y_train)

### 2. Enable Explainability

from session14_enhanced_implementation import ExplainabilityEngine

explainer = ExplainabilityEngine(model, config)
shap_results = explainer.compute_shap_values(X_test, feature_names)

### 3. Monitor in Production

from prometheus_client import start_http_server
start_http_server(8000)  # Metrics on port 8000

## Next Steps

- Read full documentation: `docs/enhanced/`
- Run examples: `examples/enhanced/`
- View API docs: `http://localhost:8014/docs`
- Grafana dashboards: `http://localhost:3000`

EOF
    
    print_success "Quick start guide created"
    
    echo ""
}

###############################################################################
# Main Deployment Flow
###############################################################################

main() {
    clear
    echo -e "${BLUE}"
    cat << 'EOF'
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║          SESSION 14 ENHANCED DEPLOYMENT                       ║
    ║          ML/VM Suite with Advanced Features                   ║
    ║                                                               ║
    ║          Version 2.0.0                                        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo ""
    
    echo "Deployment Environment: $DEPLOY_ENV"
    echo "Base Directory: $BASE_DIR"
    echo ""
    
    read -p "Continue with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled"
        exit 0
    fi
    
    # Run deployment steps
    preflight_checks
    setup_venv
    install_dependencies
    setup_database
    create_test_data
    setup_monitoring
    setup_docker
    generate_docs
    health_checks
    
    # Success message
    print_header "Deployment Complete!"
    
    echo -e "${GREEN}"
    cat << 'EOF'
    ✓ Session 14 Enhanced successfully deployed!

    Next Steps:
    ═══════════════════════════════════════════════════

    1. Activate environment:
       source venv/bin/activate

    2. Start API server:
       python -m uvicorn api.main:app --reload --port 8014

    3. Start monitoring:
       docker-compose -f docker-compose.enhanced.yml up -d prometheus grafana

    4. View API docs:
       http://localhost:8014/docs

    5. View metrics:
       http://localhost:9090 (Prometheus)
       http://localhost:3000 (Grafana, admin/admin)

    6. Run tests:
       pytest tests/test_enhanced_features.py -v

    7. Try AutoML:
       python examples/automl_quickstart.py

    Documentation:
    ═══════════════════════════════════════════════════
    - Quick Start: docs/enhanced/QUICK_START.md
    - Full README: SESSION_14_ENHANCED_README.md
    - API Reference: http://localhost:8014/docs

    Support:
    ═══════════════════════════════════════════════════
    - Email: ml-platform@company.com
    - Slack: #ml-vm-enhanced

EOF
    echo -e "${NC}"
}

# Run main deployment
main "$@"
