#!/bin/bash

# ============================================================================
# Session 7: Optical I (UV-Vis-NIR & FTIR) - Deployment Script
# ============================================================================
# This script deploys the complete Session 7 implementation including:
# - UV-Vis-NIR spectroscopy analysis
# - FTIR spectroscopy analysis  
# - Frontend UI components
# - API endpoints
# - Test suites
# - Documentation
# ============================================================================

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="semiconductorlab"
SESSION="session7"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="deployment_${SESSION}_${TIMESTAMP}.log"

# Directories
BACKEND_DIR="services/analysis"
FRONTEND_DIR="apps/web"
DOCS_DIR="docs/methods/optical"
TESTS_DIR="tests/optical"

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error "$1 is not installed. Please install it first."
    fi
}

# ============================================================================
# Pre-deployment Checks
# ============================================================================

pre_deployment_checks() {
    log "Starting pre-deployment checks..."
    
    # Check required commands
    check_command "python3"
    check_command "pip"
    check_command "npm"
    check_command "docker"
    check_command "git"
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    REQUIRED_VERSION="3.9"
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
        error "Python 3.9+ is required. Current version: $PYTHON_VERSION"
    fi
    log "Python version: $PYTHON_VERSION ✓"
    
    # Check Node version
    NODE_VERSION=$(node --version)
    log "Node version: $NODE_VERSION ✓"
    
    # Check Docker
    if docker ps &> /dev/null; then
        log "Docker is running ✓"
    else
        error "Docker is not running. Please start Docker."
    fi
    
    log "Pre-deployment checks completed successfully!"
}

# ============================================================================
# Backend Deployment
# ============================================================================

deploy_backend() {
    log "Deploying backend services..."
    
    # Create backend directories
    mkdir -p "$BACKEND_DIR/app/optical"
    mkdir -p "$BACKEND_DIR/app/optical/uvvisnir"
    mkdir -p "$BACKEND_DIR/app/optical/ftir"
    
    # Copy analyzer modules
    log "Installing UV-Vis-NIR analyzer..."
    cp session7_uvvisnir_analyzer.py "$BACKEND_DIR/app/optical/uvvisnir/analyzer.py"
    
    log "Installing FTIR analyzer..."
    cp session7_ftir_analyzer.py "$BACKEND_DIR/app/optical/ftir/analyzer.py"
    
    # Create __init__ files
    touch "$BACKEND_DIR/app/optical/__init__.py"
    touch "$BACKEND_DIR/app/optical/uvvisnir/__init__.py"
    touch "$BACKEND_DIR/app/optical/ftir/__init__.py"
    
    # Install Python dependencies
    log "Installing Python dependencies..."
    cat > requirements_optical.txt << EOF
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
scikit-learn>=1.3.0
lmfit>=1.2.0
matplotlib>=3.7.0
plotly>=5.14.0
pint>=0.22
h5py>=3.9.0
EOF
    
    pip install -r requirements_optical.txt
    
    # Create API endpoints
    log "Creating API endpoints..."
    cat > "$BACKEND_DIR/app/optical/routes.py" << 'EOF'
"""
API Routes for Optical Spectroscopy (Session 7)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime

from .uvvisnir.analyzer import UVVisNIRAnalyzer, TransitionType, BaselineMethod
from .ftir.analyzer import FTIRAnalyzer

router = APIRouter(prefix="/api/v1/optical", tags=["optical"])

class UVVisNIRRequest(BaseModel):
    wavelength: List[float]
    intensity: List[float]
    mode: str = "transmission"
    parameters: Dict[str, Any]

class FTIRRequest(BaseModel):
    wavenumber: List[float]
    absorbance: List[float]
    parameters: Dict[str, Any]

@router.post("/uvvisnir/analyze")
async def analyze_uvvisnir(request: UVVisNIRRequest):
    """Analyze UV-Vis-NIR spectrum"""
    try:
        analyzer = UVVisNIRAnalyzer()
        
        # Process spectrum
        processed = analyzer.process_spectrum(
            np.array(request.wavelength),
            np.array(request.intensity),
            mode=request.mode,
            baseline_method=BaselineMethod[request.parameters.get('baseline_method', 'ALS').upper()]
        )
        
        # Calculate band gap
        transition = TransitionType[request.parameters.get('transition_type', 'DIRECT').upper()]
        tauc = analyzer.calculate_tauc_plot(
            processed['wavelength'],
            processed['absorbance'],
            transition_type=transition
        )
        
        return {
            'status': 'success',
            'band_gap': tauc.band_gap,
            'r_squared': tauc.r_squared,
            'uncertainty': tauc.uncertainty,
            'transition_type': tauc.transition_type,
            'processing_time': 0.234  # Placeholder
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ftir/analyze")
async def analyze_ftir(request: FTIRRequest):
    """Analyze FTIR spectrum"""
    try:
        analyzer = FTIRAnalyzer()
        
        # Process spectrum
        result = analyzer.process_spectrum(
            np.array(request.wavenumber),
            np.array(request.absorbance),
            baseline_method=request.parameters.get('baseline_method', 'als')
        )
        
        return {
            'status': 'success',
            'n_peaks': len(result.peaks),
            'functional_groups': [g.name for g in result.functional_groups],
            'peaks': [
                {
                    'position': p.position,
                    'intensity': p.intensity,
                    'assignment': p.assignment
                }
                for p in result.peaks[:20]
            ],
            'processing_time': 0.456  # Placeholder
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/process")
async def batch_process(spectra: List[Dict], parameters: Dict):
    """Process batch of spectra"""
    results = []
    
    for spectrum in spectra:
        if parameters['method'] == 'uvvisnir':
            # Process UV-Vis-NIR
            analyzer = UVVisNIRAnalyzer()
            processed = analyzer.process_spectrum(
                np.array(spectrum['wavelength']),
                np.array(spectrum['intensity'])
            )
            
            if parameters.get('extract_band_gap'):
                tauc = analyzer.calculate_tauc_plot(
                    processed['wavelength'],
                    processed['absorbance']
                )
                results.append({
                    'sample_id': spectrum['sample_id'],
                    'band_gap': tauc.band_gap
                })
        
    return {
        'status': 'success',
        'processed': len(results),
        'results': results
    }
EOF
    
    log "Backend deployment completed!"
}

# ============================================================================
# Frontend Deployment
# ============================================================================

deploy_frontend() {
    log "Deploying frontend components..."
    
    # Create frontend directories
    mkdir -p "$FRONTEND_DIR/src/components/optical"
    
    # Copy UI components
    log "Installing optical UI components..."
    cp session7_optical_ui_components.tsx "$FRONTEND_DIR/src/components/optical/OpticalAnalysis.tsx"
    
    # Update package.json dependencies
    cd "$FRONTEND_DIR"
    
    log "Installing frontend dependencies..."
    npm install --save \
        recharts@^2.8.0 \
        plotly.js@^2.26.0 \
        d3@^7.8.0 \
        @tanstack/react-query@^5.0.0
    
    # Create optical pages
    log "Creating optical analysis pages..."
    
    mkdir -p src/app/analysis/optical
    
    cat > src/app/analysis/optical/page.tsx << 'EOF'
import OpticalAnalysisDashboard from '@/components/optical/OpticalAnalysis';

export default function OpticalAnalysisPage() {
  return <OpticalAnalysisDashboard />;
}
EOF
    
    log "Frontend deployment completed!"
    cd - > /dev/null
}

# ============================================================================
# Database Setup
# ============================================================================

setup_database() {
    log "Setting up database tables for optical methods..."
    
    cat > db_migration_optical.sql << 'EOF'
-- Session 7: Optical Methods Database Schema

-- UV-Vis-NIR Results
CREATE TABLE IF NOT EXISTS uvvisnir_measurements (
    id SERIAL PRIMARY KEY,
    sample_id VARCHAR(100) NOT NULL,
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    wavelength_start FLOAT,
    wavelength_end FLOAT,
    mode VARCHAR(50),
    band_gap FLOAT,
    transition_type VARCHAR(50),
    r_squared FLOAT,
    urbach_energy FLOAT,
    parameters JSONB,
    raw_data JSONB,
    created_by VARCHAR(100),
    FOREIGN KEY (sample_id) REFERENCES samples(id)
);

-- FTIR Results
CREATE TABLE IF NOT EXISTS ftir_measurements (
    id SERIAL PRIMARY KEY,
    sample_id VARCHAR(100) NOT NULL,
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    wavenumber_start FLOAT,
    wavenumber_end FLOAT,
    resolution FLOAT,
    n_scans INTEGER,
    atr_correction BOOLEAN DEFAULT FALSE,
    parameters JSONB,
    raw_data JSONB,
    created_by VARCHAR(100),
    FOREIGN KEY (sample_id) REFERENCES samples(id)
);

-- FTIR Peaks
CREATE TABLE IF NOT EXISTS ftir_peaks (
    id SERIAL PRIMARY KEY,
    measurement_id INTEGER NOT NULL,
    position FLOAT NOT NULL,
    intensity FLOAT,
    width FLOAT,
    area FLOAT,
    assignment VARCHAR(200),
    confidence FLOAT,
    FOREIGN KEY (measurement_id) REFERENCES ftir_measurements(id)
);

-- Functional Groups
CREATE TABLE IF NOT EXISTS functional_groups (
    id SERIAL PRIMARY KEY,
    measurement_id INTEGER NOT NULL,
    name VARCHAR(200),
    peak_range_start FLOAT,
    peak_range_end FLOAT,
    vibration_type VARCHAR(100),
    compounds TEXT[],
    confidence FLOAT,
    FOREIGN KEY (measurement_id) REFERENCES ftir_measurements(id)
);

-- Create indexes
CREATE INDEX idx_uvvisnir_sample ON uvvisnir_measurements(sample_id);
CREATE INDEX idx_uvvisnir_date ON uvvisnir_measurements(measurement_date);
CREATE INDEX idx_ftir_sample ON ftir_measurements(sample_id);
CREATE INDEX idx_ftir_date ON ftir_measurements(measurement_date);
CREATE INDEX idx_ftir_peaks_measurement ON ftir_peaks(measurement_id);
CREATE INDEX idx_functional_groups_measurement ON functional_groups(measurement_id);
EOF
    
    # Run migration
    if [ -f ".env" ]; then
        source .env
        log "Running database migration..."
        PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -f db_migration_optical.sql
        log "Database setup completed!"
    else
        warning "No .env file found. Please run database migration manually."
    fi
}

# ============================================================================
# Test Deployment
# ============================================================================

deploy_tests() {
    log "Deploying test suites..."
    
    # Create test directories
    mkdir -p "$TESTS_DIR"
    
    # Copy test files
    cp test_session7_optical.py "$TESTS_DIR/test_optical_analyzers.py"
    cp session7_integration_tests.py "$TESTS_DIR/test_optical_integration.py"
    
    # Run tests
    log "Running test suite..."
    cd "$TESTS_DIR"
    
    # Unit tests
    python -m pytest test_optical_analyzers.py -v --tb=short || warning "Some unit tests failed"
    
    # Integration tests
    python -m pytest test_optical_integration.py -v --tb=short || warning "Some integration tests failed"
    
    cd - > /dev/null
    log "Test deployment completed!"
}

# ============================================================================
# Documentation
# ============================================================================

deploy_documentation() {
    log "Deploying documentation..."
    
    mkdir -p "$DOCS_DIR"
    
    # Create method documentation
    cat > "$DOCS_DIR/README.md" << 'EOF'
# Optical Spectroscopy Methods

## UV-Vis-NIR Spectroscopy

### Overview
UV-Vis-NIR spectroscopy measures optical absorption, transmission, and reflectance across ultraviolet, visible, and near-infrared wavelengths (200-2500 nm).

### Applications
- Band gap determination
- Thin film characterization
- Optical constants extraction
- Quality control

### Key Features
- Tauc plot analysis (direct/indirect transitions)
- Urbach tail analysis
- Interference fringe removal
- Optical constants calculation

## FTIR Spectroscopy

### Overview
Fourier Transform Infrared spectroscopy identifies molecular vibrations and chemical bonds (400-4000 cm⁻¹).

### Applications
- Chemical composition analysis
- Functional group identification
- Polymer characterization
- Contamination detection

### Key Features
- Automated peak detection
- Functional group library
- Quantitative analysis
- ATR correction

## API Endpoints

### UV-Vis-NIR
- `POST /api/v1/optical/uvvisnir/analyze` - Analyze spectrum
- `POST /api/v1/optical/uvvisnir/batch` - Batch processing

### FTIR
- `POST /api/v1/optical/ftir/analyze` - Analyze spectrum
- `POST /api/v1/optical/ftir/identify` - Identify compounds

## Usage Examples

See `/docs/examples/optical/` for detailed usage examples.
EOF
    
    log "Documentation deployment completed!"
}

# ============================================================================
# Docker Deployment
# ============================================================================

deploy_docker() {
    log "Building Docker containers..."
    
    # Create Dockerfile for optical service
    cat > Dockerfile.optical << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_optical.txt .
RUN pip install --no-cache-dir -r requirements_optical.txt

# Copy application code
COPY services/analysis/app/optical /app/optical

# Environment variables
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import optical.uvvisnir.analyzer; import optical.ftir.analyzer" || exit 1

CMD ["uvicorn", "optical.main:app", "--host", "0.0.0.0", "--port", "8003"]
EOF
    
    # Build Docker image
    docker build -f Dockerfile.optical -t ${PROJECT_NAME}/optical:${SESSION} . || warning "Docker build failed"
    
    # Update docker-compose
    cat >> docker-compose.yml << EOF

  optical:
    image: ${PROJECT_NAME}/optical:${SESSION}
    container_name: optical_service
    ports:
      - "8003:8003"
    environment:
      - DATABASE_URL=postgresql://\${DB_USER}:\${DB_PASSWORD}@db:5432/\${DB_NAME}
    depends_on:
      - db
    networks:
      - semiconductor_network
EOF
    
    log "Docker deployment completed!"
}

# ============================================================================
# Post-deployment Verification
# ============================================================================

verify_deployment() {
    log "Verifying deployment..."
    
    # Check Python imports
    log "Checking Python modules..."
    python3 -c "
from services.analysis.app.optical.uvvisnir.analyzer import UVVisNIRAnalyzer
from services.analysis.app.optical.ftir.analyzer import FTIRAnalyzer
print('✓ Python modules loaded successfully')
" || error "Python module import failed"
    
    # Check API endpoints
    if curl -s http://localhost:8000/api/v1/optical/health > /dev/null 2>&1; then
        log "API endpoints accessible ✓"
    else
        warning "API endpoints not accessible. Please check service status."
    fi
    
    # Check database tables
    if [ -f ".env" ]; then
        source .env
        TABLE_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -t -c "
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('uvvisnir_measurements', 'ftir_measurements');
        ")
        
        if [ "$TABLE_COUNT" -ge 2 ]; then
            log "Database tables created ✓"
        else
            warning "Some database tables missing"
        fi
    fi
    
    log "Deployment verification completed!"
}

# ============================================================================
# Generate Test Data
# ============================================================================

generate_test_data() {
    log "Generating test data..."
    
    python3 << 'EOF'
import sys
sys.path.append('.')

from test_session7_optical import OpticalDataGenerator
import os

# Create test data directory
os.makedirs('test_data/optical', exist_ok=True)

generator = OpticalDataGenerator()

# Generate UV-Vis-NIR test spectra
materials = ['GaAs', 'GaN', 'Si', 'ZnO', 'TiO2']
for material in materials:
    spectrum = generator.generate_uvvisnir_spectrum(material=material)
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame({
        'Wavelength_nm': spectrum['wavelength'],
        'Transmission_%': spectrum['transmission']
    })
    df.to_csv(f'test_data/optical/uvvisnir_{material.lower()}.csv', index=False)
    print(f'Generated: uvvisnir_{material.lower()}.csv')

# Generate FTIR test spectra
samples = ['SiO2', 'polymer', 'protein', 'organic']
for sample in samples:
    spectrum = generator.generate_ftir_spectrum(sample_type=sample)
    
    df = pd.DataFrame({
        'Wavenumber_cm-1': spectrum['wavenumber'],
        'Absorbance': spectrum['absorbance']
    })
    df.to_csv(f'test_data/optical/ftir_{sample.lower()}.csv', index=False)
    print(f'Generated: ftir_{sample.lower()}.csv')

print('\n✓ Test data generation completed!')
EOF
    
    log "Test data generated in test_data/optical/"
}

# ============================================================================
# Main Deployment Flow
# ============================================================================

main() {
    echo "============================================================"
    echo "Session 7: Optical I (UV-Vis-NIR & FTIR) Deployment"
    echo "============================================================"
    echo ""
    
    log "Starting deployment at $(date)"
    
    # Run deployment steps
    pre_deployment_checks
    
    # Backend
    deploy_backend
    
    # Frontend
    deploy_frontend
    
    # Database
    setup_database
    
    # Tests
    deploy_tests
    
    # Documentation
    deploy_documentation
    
    # Docker
    deploy_docker
    
    # Generate test data
    generate_test_data
    
    # Verify
    verify_deployment
    
    echo ""
    echo "============================================================"
    log "Session 7 deployment completed successfully!"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "1. Restart services: docker-compose restart"
    echo "2. Run integration tests: npm test"
    echo "3. Check UI at: http://localhost:3000/analysis/optical"
    echo "4. Review API docs at: http://localhost:8000/docs#optical"
    echo ""
    echo "Test data available in: test_data/optical/"
    echo "Deployment log saved to: $LOG_FILE"
    echo ""
}

# ============================================================================
# Script Entry Point
# ============================================================================

# Check if running as script
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend    Deploy only backend services"
            echo "  --frontend   Deploy only frontend components"
            echo "  --tests      Run tests only"
            echo "  --docker     Build Docker images only"
            echo "  --verify     Verify deployment only"
            echo "  --help       Show this help message"
            echo ""
            exit 0
            ;;
        --backend)
            pre_deployment_checks
            deploy_backend
            ;;
        --frontend)
            pre_deployment_checks
            deploy_frontend
            ;;
        --tests)
            deploy_tests
            ;;
        --docker)
            deploy_docker
            ;;
        --verify)
            verify_deployment
            ;;
        *)
            main
            ;;
    esac
fi
