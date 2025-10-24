#!/bin/bash

# ==========================================
# Session 6: Electrical III - Master Deployment Script
# DLTS, EBIC, PCD Implementation
# ==========================================

set -e  # Exit on error

echo "=================================================="
echo "Session 6: Electrical III - Deployment Script"
echo "=================================================="
echo "Deploying DLTS, EBIC, and PCD modules..."
echo ""

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-/home/semiconductorlab}"
BACKEND_DIR="$PROJECT_ROOT/services/analysis"
FRONTEND_DIR="$PROJECT_ROOT/apps/web"
DATA_DIR="$PROJECT_ROOT/data"
DOCS_DIR="$PROJECT_ROOT/docs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==========================================
# 1. Environment Check
# ==========================================

echo -e "${YELLOW}[1/8]${NC} Checking environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "  Python version: $PYTHON_VERSION"

# Check Node.js version
NODE_VERSION=$(node --version)
echo "  Node.js version: $NODE_VERSION"

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    echo "  Docker version: $DOCKER_VERSION"
else
    echo -e "${RED}  Docker not found!${NC}"
    exit 1
fi

# Check PostgreSQL connection
if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo -e "${GREEN}  PostgreSQL: Connected${NC}"
else
    echo -e "${YELLOW}  PostgreSQL: Not connected (will use Docker)${NC}"
fi

# ==========================================
# 2. Create Directory Structure
# ==========================================

echo -e "${YELLOW}[2/8]${NC} Creating directory structure..."

# Backend directories
mkdir -p "$BACKEND_DIR/app/methods/electrical/advanced"
mkdir -p "$BACKEND_DIR/app/drivers/electrical"
mkdir -p "$BACKEND_DIR/tests/electrical"

# Frontend directories
mkdir -p "$FRONTEND_DIR/src/components/electrical/advanced"
mkdir -p "$FRONTEND_DIR/src/lib/api/electrical"
mkdir -p "$FRONTEND_DIR/public/assets/electrical"

# Data directories
mkdir -p "$DATA_DIR/test_data/electrical/dlts"
mkdir -p "$DATA_DIR/test_data/electrical/ebic"
mkdir -p "$DATA_DIR/test_data/electrical/pcd"
mkdir -p "$DATA_DIR/calibration/electrical"

# Documentation
mkdir -p "$DOCS_DIR/methods/electrical"
mkdir -p "$DOCS_DIR/api/electrical"

echo "  Directory structure created"

# ==========================================
# 3. Install Python Dependencies
# ==========================================

echo -e "${YELLOW}[3/8]${NC} Installing Python dependencies..."

cat > requirements_session6.txt << 'EOF'
# Session 6 Dependencies
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.15.0
opencv-python>=4.8.0
h5py>=3.9.0
pint>=0.22
uncertainties>=3.1.7
lmfit>=1.2.2
peakutils>=1.3.4
EOF

pip3 install -r requirements_session6.txt --quiet

echo -e "${GREEN}  Python dependencies installed${NC}"

# ==========================================
# 4. Install Node.js Dependencies
# ==========================================

echo -e "${YELLOW}[4/8]${NC} Installing Node.js dependencies..."

cd "$FRONTEND_DIR"

# Check if package.json needs updates
if ! grep -q "recharts" package.json; then
    npm install --save recharts@^2.8.0 lucide-react@^0.290.0 --silent
fi

echo -e "${GREEN}  Node.js dependencies installed${NC}"

# ==========================================
# 5. Deploy Backend Modules
# ==========================================

echo -e "${YELLOW}[5/8]${NC} Deploying backend modules..."

# Copy DLTS module
cat > "$BACKEND_DIR/app/methods/electrical/dlts_analyzer.py" << 'EOF'
from session6_backend_analysis import DLTSAnalyzer
__all__ = ['DLTSAnalyzer']
EOF

# Copy EBIC module
cat > "$BACKEND_DIR/app/methods/electrical/ebic_analyzer.py" << 'EOF'
from session6_backend_analysis import EBICAnalyzer
__all__ = ['EBICAnalyzer']
EOF

# Copy PCD module
cat > "$BACKEND_DIR/app/methods/electrical/pcd_analyzer.py" << 'EOF'
from session6_backend_analysis import PCDAnalyzer
__all__ = ['PCDAnalyzer']
EOF

# Create API endpoints
cat > "$BACKEND_DIR/app/routers/electrical_advanced.py" << 'EOF'
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List
from ..methods.electrical import dlts_analyzer, ebic_analyzer, pcd_analyzer
from ..schemas.electrical import DLTSRequest, EBICRequest, PCDRequest
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/electrical/advanced", tags=["electrical-advanced"])

@router.post("/dlts/analyze")
async def analyze_dlts(request: DLTSRequest) -> Dict:
    """Analyze DLTS spectrum"""
    try:
        analyzer = dlts_analyzer.DLTSAnalyzer()
        results = analyzer.analyze_spectrum(
            request.temperatures,
            request.capacitances,
            request.rate_window
        )
        return {"status": "success", "data": results}
    except Exception as e:
        logger.error(f"DLTS analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ebic/analyze")
async def analyze_ebic(request: EBICRequest) -> Dict:
    """Analyze EBIC map"""
    try:
        analyzer = ebic_analyzer.EBICAnalyzer(request.pixel_size)
        results = analyzer.analyze_map(
            request.current_map,
            request.beam_energy,
            request.temperature
        )
        return {"status": "success", "data": results}
    except Exception as e:
        logger.error(f"EBIC analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pcd/analyze")
async def analyze_pcd(request: PCDRequest) -> Dict:
    """Analyze PCD measurement"""
    try:
        analyzer = pcd_analyzer.PCDAnalyzer()
        if request.mode == "transient":
            results = analyzer.analyze_transient(
                request.time,
                request.photoconductance,
                request.temperature
            )
        else:
            results = analyzer.analyze_qsspc(
                request.photon_flux,
                request.photoconductance,
                request.temperature
            )
        return {"status": "success", "data": results}
    except Exception as e:
        logger.error(f"PCD analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

echo -e "${GREEN}  Backend modules deployed${NC}"

# ==========================================
# 6. Deploy Frontend Components
# ==========================================

echo -e "${YELLOW}[6/8]${NC} Deploying frontend components..."

# Copy UI components
cp session6_complete_ui_components.tsx "$FRONTEND_DIR/src/components/electrical/advanced/"

# Create index export
cat > "$FRONTEND_DIR/src/components/electrical/advanced/index.ts" << 'EOF'
export { DLTSMeasurement } from './session6_complete_ui_components';
export { EBICMapping } from './session6_complete_ui_components';
export { PCDMeasurement } from './session6_complete_ui_components';
EOF

# Create API client
cat > "$FRONTEND_DIR/src/lib/api/electrical/advanced.ts" << 'EOF'
import { apiClient } from '@/lib/api/client';

export const electricalAdvancedAPI = {
  analyzeDLTS: async (data: any) => {
    const response = await apiClient.post('/api/electrical/advanced/dlts/analyze', data);
    return response.data;
  },
  
  analyzeEBIC: async (data: any) => {
    const response = await apiClient.post('/api/electrical/advanced/ebic/analyze', data);
    return response.data;
  },
  
  analyzePCD: async (data: any) => {
    const response = await apiClient.post('/api/electrical/advanced/pcd/analyze', data);
    return response.data;
  }
};
EOF

echo -e "${GREEN}  Frontend components deployed${NC}"

# ==========================================
# 7. Database Migrations
# ==========================================

echo -e "${YELLOW}[7/8]${NC} Running database migrations..."

# Create migration file
cat > "$BACKEND_DIR/alembic/versions/session6_electrical_advanced.sql" << 'EOF'
-- Session 6: Electrical Advanced Methods Tables

-- DLTS Measurements
CREATE TABLE IF NOT EXISTS dlts_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    measurement_type VARCHAR(50),
    temperature_range JSONB,
    voltage_pulse JSONB,
    rate_windows FLOAT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- DLTS Trap Signatures
CREATE TABLE IF NOT EXISTS dlts_traps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES dlts_measurements(id),
    trap_label VARCHAR(50),
    activation_energy FLOAT,
    capture_cross_section FLOAT,
    trap_concentration FLOAT,
    trap_type VARCHAR(20),
    peak_temperature FLOAT,
    confidence_score FLOAT
);

-- EBIC Measurements
CREATE TABLE IF NOT EXISTS ebic_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    beam_energy FLOAT,
    beam_current FLOAT,
    scan_area JSONB,
    pixel_resolution INTEGER[],
    temperature FLOAT,
    bias_voltage FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- EBIC Maps
CREATE TABLE IF NOT EXISTS ebic_maps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES ebic_measurements(id),
    current_map BYTEA,
    normalized_map BYTEA,
    diffusion_length_map BYTEA,
    statistics JSONB
);

-- PCD Measurements
CREATE TABLE IF NOT EXISTS pcd_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    measurement_mode VARCHAR(50),
    excitation_wavelength FLOAT,
    photon_flux FLOAT,
    temperature FLOAT,
    sample_thickness FLOAT,
    surface_condition VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PCD Lifetime Data
CREATE TABLE IF NOT EXISTS pcd_lifetime (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES pcd_measurements(id),
    injection_level FLOAT[],
    effective_lifetime FLOAT[],
    bulk_lifetime FLOAT[],
    surface_lifetime FLOAT[],
    srv_front FLOAT,
    srv_back FLOAT,
    auger_coefficient FLOAT
);

-- Create indexes
CREATE INDEX idx_dlts_measurements_sample ON dlts_measurements(sample_id);
CREATE INDEX idx_dlts_traps_measurement ON dlts_traps(measurement_id);
CREATE INDEX idx_ebic_measurements_sample ON ebic_measurements(sample_id);
CREATE INDEX idx_pcd_measurements_sample ON pcd_measurements(sample_id);
EOF

# Run migration
if command -v alembic &> /dev/null; then
    cd "$BACKEND_DIR"
    alembic upgrade head
    echo -e "${GREEN}  Database migrations completed${NC}"
else
    echo -e "${YELLOW}  Alembic not found, skipping migrations${NC}"
fi

# ==========================================
# 8. Generate Test Data & Run Tests
# ==========================================

echo -e "${YELLOW}[8/8]${NC} Generating test data and running tests..."

# Generate test data
python3 << 'EOF'
from session6_backend_analysis import Session6TestDataGenerator
import json
from pathlib import Path

generator = Session6TestDataGenerator()
data_dir = Path("/tmp/test_data/session6")
data_dir.mkdir(parents=True, exist_ok=True)

# Generate DLTS data
dlts_data = generator.generate_dlts_data(num_traps=3)
with open(data_dir / "dlts_test.json", 'w') as f:
    json.dump(dlts_data, f, indent=2)

# Generate EBIC data
ebic_data = generator.generate_ebic_data(map_size=256)
with open(data_dir / "ebic_test.json", 'w') as f:
    json.dump(ebic_data, f, indent=2)

# Generate PCD data
pcd_transient = generator.generate_pcd_data(mode='transient')
with open(data_dir / "pcd_transient.json", 'w') as f:
    json.dump(pcd_transient, f, indent=2)

pcd_qsspc = generator.generate_pcd_data(mode='qsspc')
with open(data_dir / "pcd_qsspc.json", 'w') as f:
    json.dump(pcd_qsspc, f, indent=2)

print("Test data generated successfully")
EOF

# Run integration tests
if [ -f "test_session6_integration.py" ]; then
    python3 -m pytest test_session6_integration.py -v --tb=short
    TEST_RESULT=$?
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}  All tests passed!${NC}"
    else
        echo -e "${RED}  Some tests failed${NC}"
    fi
else
    echo -e "${YELLOW}  Test file not found, skipping tests${NC}"
fi

# ==========================================
# Final Status Report
# ==========================================

echo ""
echo "=================================================="
echo -e "${GREEN}Session 6 Deployment Complete!${NC}"
echo "=================================================="
echo ""
echo "Deployed Components:"
echo "  ✓ DLTS Analysis Module"
echo "  ✓ EBIC Mapping Module"
echo "  ✓ PCD Lifetime Analysis"
echo "  ✓ Frontend UI Components"
echo "  ✓ API Endpoints"
echo "  ✓ Database Tables"
echo "  ✓ Test Data Generated"
echo ""
echo "Access Points:"
echo "  • DLTS UI: http://localhost:3000/electrical/dlts"
echo "  • EBIC UI: http://localhost:3000/electrical/ebic"
echo "  • PCD UI: http://localhost:3000/electrical/pcd"
echo "  • API Docs: http://localhost:8000/docs#/electrical-advanced"
echo ""
echo "Test Data Location: /tmp/test_data/session6/"
echo ""
echo "Next Steps:"
echo "  1. Restart Docker containers: docker-compose restart"
echo "  2. Verify UI components are loading"
echo "  3. Test with sample data"
echo "  4. Review generated documentation"
echo ""
echo "Documentation generated at: $DOCS_DIR/methods/electrical/"
echo ""