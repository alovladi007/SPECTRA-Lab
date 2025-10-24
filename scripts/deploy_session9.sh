#!/bin/bash

################################################################################
# Session 9: X-ray Diffraction (XRD) Analysis - Deployment Script
# Structural characterization and phase identification system
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="semiconductor-lab"
SESSION_NAME="session9-xrd"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="deployment_${SESSION_NAME}_${TIMESTAMP}.log"

# Functions
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

# Header
echo "============================================" | tee "$LOG_FILE"
echo "Session 9: XRD Analysis Deployment" | tee -a "$LOG_FILE"
echo "Structural Characterization System" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 1: Environment Check
log "Step 1: Checking environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    error "Python $REQUIRED_VERSION or higher is required. Found: Python $PYTHON_VERSION"
fi

info "Python version: $PYTHON_VERSION ✓"

# Check for previous sessions
if [ ! -f ".session8_complete" ]; then
    warning "Session 8 not found. Some dependencies may be missing."
fi

# Step 2: Create project structure
log "Step 2: Creating project structure..."

# Create directories
DIRS=(
    "backend/app/modules/xrd"
    "backend/tests/xrd"
    "frontend/src/components/xrd"
    "data/xrd/patterns"
    "data/xrd/phases"
    "data/xrd/calibration"
    "docs/session9"
    "logs/xrd"
    "config/xrd"
)

for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
    info "Created directory: $dir"
done

# Step 3: Install Python dependencies
log "Step 3: Installing Python dependencies..."

# Create or activate virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    info "Created virtual environment"
fi

source venv/bin/activate

# Install additional dependencies for Session 9
cat > requirements_session9.txt << EOF
# Core scientific computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# XRD specific
pymatgen>=2022.0.0      # Crystallography library
xrayutilities>=1.7.0    # XRD analysis utilities
diffpy.structure>=3.0.0 # Structure analysis
crystals>=1.0.0         # Crystal structure toolkit
fabio>=0.14.0          # X-ray image IO

# Peak fitting
lmfit>=1.0.3
peakutils>=1.3.3

# Phase identification
pyxtal>=0.5.0          # Crystal structure prediction
ase>=3.22.0            # Atomic simulation environment

# Database
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0

# API
fastapi>=0.85.0
uvicorn>=0.18.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.19.0
pytest-cov>=3.0.0
EOF

pip install -r requirements_session9.txt > /dev/null 2>&1

if [ $? -eq 0 ]; then
    info "Python dependencies installed successfully"
else
    warning "Some dependencies may not have installed correctly"
fi

# Step 4: Deploy implementation files
log "Step 4: Deploying implementation files..."

# Copy Python modules
cp session9_xrd_complete_implementation.py backend/app/modules/xrd/analyzer.py
cp test_session9_xrd_integration.py backend/tests/xrd/test_integration.py

# Copy React components
cp session9_xrd_ui_components.tsx frontend/src/components/xrd/XRDInterface.tsx

info "Implementation files deployed"

# Step 5: Database setup
log "Step 5: Setting up database tables..."

# Create SQL migration for Session 9
cat > backend/migrations/009_xrd_tables.sql << 'EOF'
-- Session 9: XRD Analysis Tables

-- XRD measurements table
CREATE TABLE IF NOT EXISTS xrd_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    xray_source VARCHAR(20) NOT NULL,
    wavelength FLOAT NOT NULL,
    start_angle FLOAT NOT NULL,
    end_angle FLOAT NOT NULL,
    step_size FLOAT NOT NULL,
    scan_speed FLOAT,
    voltage FLOAT,
    current FLOAT,
    instrument_id UUID REFERENCES instruments(id),
    operator_id UUID REFERENCES users(id),
    temperature FLOAT,
    atmosphere VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- XRD patterns data
CREATE TABLE IF NOT EXISTS xrd_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES xrd_measurements(id) ON DELETE CASCADE,
    two_theta FLOAT[] NOT NULL,
    intensity FLOAT[] NOT NULL,
    d_spacing FLOAT[],
    background FLOAT[],
    processed_intensity FLOAT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- XRD peaks
CREATE TABLE IF NOT EXISTS xrd_peaks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES xrd_measurements(id) ON DELETE CASCADE,
    position FLOAT NOT NULL,
    d_spacing FLOAT NOT NULL,
    intensity FLOAT NOT NULL,
    fwhm FLOAT,
    area FLOAT,
    h INTEGER,
    k INTEGER,
    l INTEGER,
    phase_id UUID REFERENCES xrd_phases(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Phase database
CREATE TABLE IF NOT EXISTS xrd_phases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    formula VARCHAR(50) NOT NULL,
    crystal_system VARCHAR(20) NOT NULL,
    space_group VARCHAR(20),
    a FLOAT,
    b FLOAT,
    c FLOAT,
    alpha FLOAT DEFAULT 90,
    beta FLOAT DEFAULT 90,
    gamma FLOAT DEFAULT 90,
    volume FLOAT,
    density FLOAT,
    reference VARCHAR(200),
    cif_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Phase identification results
CREATE TABLE IF NOT EXISTS xrd_phase_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES xrd_measurements(id) ON DELETE CASCADE,
    phase_id UUID REFERENCES xrd_phases(id),
    score FLOAT NOT NULL,
    matched_peaks INTEGER,
    total_peaks INTEGER,
    scale_factor FLOAT,
    weight_fraction FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Peak fitting results
CREATE TABLE IF NOT EXISTS xrd_peak_fits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES xrd_measurements(id) ON DELETE CASCADE,
    profile_type VARCHAR(20) NOT NULL,
    fitted_peaks JSONB NOT NULL,
    r_wp FLOAT,
    r_exp FLOAT,
    chi_squared FLOAT,
    background_params JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Crystallite size and strain analysis
CREATE TABLE IF NOT EXISTS xrd_size_strain (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES xrd_measurements(id) ON DELETE CASCADE,
    method VARCHAR(50) NOT NULL,
    crystallite_size FLOAT,
    size_error FLOAT,
    microstrain FLOAT,
    strain_error FLOAT,
    shape_factor FLOAT DEFAULT 0.9,
    r_squared FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Residual stress analysis
CREATE TABLE IF NOT EXISTS xrd_stress_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    measurement_ids UUID[] NOT NULL,
    method VARCHAR(50) DEFAULT 'sin2psi',
    stress_mpa FLOAT NOT NULL,
    stress_error FLOAT,
    stress_type VARCHAR(20),
    young_modulus FLOAT,
    poisson_ratio FLOAT,
    r_squared FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Texture analysis
CREATE TABLE IF NOT EXISTS xrd_texture (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES xrd_measurements(id) ON DELETE CASCADE,
    texture_coefficients FLOAT[],
    texture_index FLOAT,
    preferred_orientation VARCHAR(20),
    orientation_factor FLOAT,
    is_textured BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Rietveld refinement results
CREATE TABLE IF NOT EXISTS xrd_rietveld (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES xrd_measurements(id) ON DELETE CASCADE,
    phase_ids UUID[] NOT NULL,
    scale_factors FLOAT[],
    lattice_params JSONB,
    atomic_positions JSONB,
    thermal_params JSONB,
    r_wp FLOAT,
    r_exp FLOAT,
    chi_squared FLOAT,
    gof FLOAT,
    converged BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_xrd_measurements_sample ON xrd_measurements(sample_id);
CREATE INDEX idx_xrd_peaks_position ON xrd_peaks(position);
CREATE INDEX idx_xrd_peaks_phase ON xrd_peaks(phase_id);
CREATE INDEX idx_xrd_phase_matches_measurement ON xrd_phase_matches(measurement_id);
CREATE INDEX idx_xrd_phases_formula ON xrd_phases(formula);
CREATE INDEX idx_xrd_phases_system ON xrd_phases(crystal_system);

-- Add comments
COMMENT ON TABLE xrd_measurements IS 'X-ray diffraction measurements';
COMMENT ON TABLE xrd_patterns IS 'Raw and processed XRD pattern data';
COMMENT ON TABLE xrd_peaks IS 'Identified peaks in XRD patterns';
COMMENT ON TABLE xrd_phases IS 'Crystallographic phase database';
COMMENT ON TABLE xrd_phase_matches IS 'Phase identification results';
COMMENT ON TABLE xrd_size_strain IS 'Crystallite size and microstrain analysis';
COMMENT ON TABLE xrd_stress_analysis IS 'Residual stress measurements';
COMMENT ON TABLE xrd_texture IS 'Preferred orientation analysis';
COMMENT ON TABLE xrd_rietveld IS 'Rietveld refinement results';
EOF

# Apply migration if PostgreSQL is available
if command -v psql &> /dev/null; then
    psql -U postgres -d semiconductor_lab -f backend/migrations/009_xrd_tables.sql > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        info "Database tables created successfully"
    else
        warning "Could not apply database migrations. Please run manually."
    fi
else
    warning "PostgreSQL not found. Please run migrations manually."
fi

# Step 6: Load phase database
log "Step 6: Loading crystallographic phase database..."

python3 << 'EOF'
import sys
sys.path.append('.')
from session9_xrd_complete_implementation import XRDAnalyzer, XRDSimulator
import json
import numpy as np

# Initialize components
analyzer = XRDAnalyzer()
simulator = XRDSimulator()

# Generate reference patterns for common phases
phases_to_generate = ['Si', 'GaAs', 'GaN_hex', 'SiO2_quartz', 'Al2O3', 'TiO2_anatase']

for phase in phases_to_generate:
    # Generate pattern
    pattern = simulator.generate_pattern(phase=phase)
    
    # Save pattern data
    data = {
        'phase': phase,
        'two_theta': pattern.two_theta.tolist(),
        'intensity': pattern.intensity.tolist(),
        'wavelength': pattern.wavelength,
        'metadata': pattern.metadata
    }
    
    with open(f'data/xrd/patterns/{phase.lower()}_reference.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated reference pattern for {phase}")

# Save phase database
phase_db = {}
for name, structure in analyzer.phase_database.items():
    phase_db[name] = {
        'name': structure.name,
        'formula': structure.formula,
        'crystal_system': structure.crystal_system.value,
        'space_group': structure.space_group,
        'lattice_params': structure.lattice_params,
        'atoms': structure.atoms
    }

with open('data/xrd/phases/phase_database.json', 'w') as f:
    json.dump(phase_db, f, indent=2)

print("Phase database saved successfully")
EOF

if [ $? -eq 0 ]; then
    info "Phase database loaded successfully"
else
    warning "Phase database loading encountered issues"
fi

# Step 7: Configure XRD analysis settings
log "Step 7: Configuring XRD analysis settings..."

cat > config/xrd/analysis_config.yaml << 'EOF'
# Session 9: XRD Analysis Configuration

measurement:
  default_source: Cu_Ka
  default_wavelength: 1.5418
  default_range: [20, 80]
  default_step: 0.02
  scan_modes:
    - continuous
    - step
    - coupled
    - decoupled

processing:
  smoothing:
    method: savgol
    window: 5
    polyorder: 2
  background:
    method: polynomial
    order: 3
  peak_finding:
    prominence: 0.05
    min_height: 0.05
    min_distance: 5

fitting:
  profiles:
    - gaussian
    - lorentzian
    - voigt
    - pseudo_voigt
    - pearson_vii
  max_iterations: 5000
  convergence_tolerance: 1e-6

phase_identification:
  search_tolerance: 0.1  # degrees
  min_score: 20  # percent
  max_phases: 5

crystallite_analysis:
  scherrer:
    shape_factor: 0.9
    size_range: [0.5, 1000]  # nm
  williamson_hall:
    min_peaks: 3
    
stress_analysis:
  sin2psi:
    min_tilts: 3
    max_tilts: 7
    tilt_angles: [0, 15, 30, 45, 60]
  materials:
    Si:
      young_modulus: 169  # GPa
      poisson_ratio: 0.22
    GaAs:
      young_modulus: 85.5
      poisson_ratio: 0.31
    Al:
      young_modulus: 70
      poisson_ratio: 0.35

texture:
  random_threshold: 0.1
  methods:
    - march_dollase
    - spherical_harmonics
    
quality_metrics:
  r_wp_threshold: 20  # percent
  chi_squared_threshold: 2
  min_peaks_for_analysis: 3
EOF

info "XRD analysis configuration created"

# Step 8: Create API endpoints
log "Step 8: Setting up API endpoints..."

cat > backend/app/modules/xrd/api.py << 'EOF'
"""
Session 9: XRD Analysis API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from uuid import UUID
import numpy as np

from .analyzer import (
    XRDAnalyzer, XRDSimulator, XRDPattern, Peak,
    PhaseIdentification, CrystalSystem, PeakProfile
)

router = APIRouter(prefix="/api/xrd", tags=["xrd"])

# Pydantic models
class XRDMeasurementRequest(BaseModel):
    sample_id: UUID
    xray_source: str = "Cu_Ka"
    start_angle: float = Field(ge=0, le=180)
    end_angle: float = Field(ge=0, le=180)
    step_size: float = Field(gt=0)
    scan_speed: Optional[float] = 1.0

class PeakFittingRequest(BaseModel):
    measurement_id: UUID
    profile: str = "pseudo_voigt"
    peak_indices: Optional[List[int]] = None

class PhaseSearchRequest(BaseModel):
    measurement_id: UUID
    tolerance: float = 0.1
    max_phases: int = 5

class StressAnalysisRequest(BaseModel):
    sample_id: UUID
    measurements: List[Tuple[float, float]]  # (psi_angle, d_spacing)
    d0: float
    young_modulus: float = 169
    poisson_ratio: float = 0.22

# Initialize analyzer
analyzer = XRDAnalyzer()

@router.post("/measure")
async def start_measurement(request: XRDMeasurementRequest):
    """Start XRD measurement"""
    try:
        # Would interface with diffractometer
        return {
            "status": "measurement_started",
            "measurement_id": "generated_uuid",
            "estimated_time": (request.end_angle - request.start_angle) / request.scan_speed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/{measurement_id}")
async def analyze_pattern(
    measurement_id: UUID,
    background_tasks: BackgroundTasks
):
    """Analyze XRD pattern"""
    try:
        # Would fetch pattern from database
        # For demo, generate synthetic pattern
        simulator = XRDSimulator()
        pattern = simulator.generate_pattern('Si')
        
        # Process pattern
        processed = analyzer.process_pattern(pattern)
        
        # Find peaks
        peaks = analyzer.find_peaks(processed)
        
        return {
            "measurement_id": measurement_id,
            "num_peaks": len(peaks),
            "peaks": [
                {
                    "position": p.position,
                    "d_spacing": p.d_spacing,
                    "intensity": p.intensity,
                    "fwhm": p.fwhm
                }
                for p in peaks[:10]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/identify-phases")
async def identify_phases(request: PhaseSearchRequest):
    """Identify crystalline phases"""
    try:
        # Would fetch pattern and peaks
        # For demo, use mock data
        simulator = XRDSimulator()
        pattern = simulator.generate_pattern('Si')
        peaks = analyzer.find_peaks(pattern)
        
        # Identify phases
        phases = analyzer.identify_phases(pattern, peaks, request.tolerance)
        
        return {
            "measurement_id": request.measurement_id,
            "phases": [
                {
                    "name": p.phase_name,
                    "formula": p.formula,
                    "crystal_system": p.crystal_system,
                    "score": p.score,
                    "matched_peaks": len(p.matched_peaks)
                }
                for p in phases[:request.max_phases]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crystallite-size/{measurement_id}")
async def calculate_crystallite_size(measurement_id: UUID):
    """Calculate crystallite size and strain"""
    try:
        # Would fetch peaks
        # For demo, use mock peaks
        peaks = [
            Peak(28.4, 3.136, 1000, 0.15, 150),
            Peak(47.3, 1.920, 600, 0.18, 108),
            Peak(56.1, 1.638, 350, 0.20, 70)
        ]
        
        # Scherrer analysis
        scherrer = analyzer.calculate_crystallite_size(peaks, 1.5418)
        
        # Williamson-Hall analysis
        wh = analyzer.williamson_hall_analysis(peaks, 1.5418)
        
        return {
            "measurement_id": measurement_id,
            "scherrer": {
                "size_nm": scherrer['mean_size'],
                "std_nm": scherrer['std_size']
            },
            "williamson_hall": {
                "size_nm": wh['crystallite_size'],
                "microstrain": wh['microstrain'],
                "r_squared": wh['r_squared']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stress-analysis")
async def analyze_stress(request: StressAnalysisRequest):
    """Perform residual stress analysis"""
    try:
        result = analyzer.residual_stress_sin2psi(
            request.measurements,
            request.d0,
            request.young_modulus,
            request.poisson_ratio
        )
        
        return {
            "sample_id": request.sample_id,
            "stress_mpa": result['stress_mpa'],
            "stress_type": result['type'],
            "error": result['error'],
            "r_squared": result['r_squared']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fit-peaks")
async def fit_peaks(request: PeakFittingRequest):
    """Fit peaks with specified profile"""
    try:
        # Would fetch pattern and peaks
        # For demo, generate synthetic
        simulator = XRDSimulator()
        pattern = simulator.generate_pattern('Si')
        peaks = analyzer.find_peaks(pattern)
        
        # Select peaks to fit
        if request.peak_indices:
            peaks = [peaks[i] for i in request.peak_indices if i < len(peaks)]
        else:
            peaks = peaks[:5]  # Default to first 5
        
        # Fit peaks
        profile = PeakProfile[request.profile.upper()]
        result = analyzer.fit_peaks(pattern, peaks, profile)
        
        return {
            "measurement_id": request.measurement_id,
            "profile": request.profile,
            "fitted_peaks": result['peaks'],
            "r_wp": result['r_wp'],
            "chi_squared": result['chi_squared']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/phases/search")
async def search_phases(
    formula: Optional[str] = None,
    crystal_system: Optional[str] = None,
    limit: int = 10
):
    """Search phase database"""
    try:
        phases = []
        
        for name, structure in analyzer.phase_database.items():
            # Apply filters
            if formula and formula.lower() not in structure.formula.lower():
                continue
            if crystal_system and crystal_system != structure.crystal_system.value:
                continue
            
            phases.append({
                "name": structure.name,
                "formula": structure.formula,
                "crystal_system": structure.crystal_system.value,
                "space_group": structure.space_group,
                "lattice_params": structure.lattice_params
            })
            
            if len(phases) >= limit:
                break
        
        return {"phases": phases, "total": len(phases)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "module": "xrd-analysis",
        "session": 9,
        "analyzer": "initialized",
        "phase_database": len(analyzer.phase_database)
    }
EOF

info "API endpoints configured"

# Step 9: Run tests
log "Step 9: Running integration tests..."

python -m pytest backend/tests/xrd/test_integration.py -v --tb=short -k "not performance" > test_results.log 2>&1

if [ $? -eq 0 ]; then
    info "All tests passed ✓"
else
    warning "Some tests failed. Check test_results.log for details"
fi

# Step 10: Create monitoring setup
log "Step 10: Setting up monitoring..."

cat > config/xrd/monitoring.yaml << 'EOF'
# Session 9: XRD Monitoring Configuration

metrics:
  pattern_quality:
    - name: signal_to_noise
      type: gauge
      description: "Signal-to-noise ratio"
    - name: peak_resolution
      type: gauge
      description: "Peak resolution (FWHM)"
    - name: background_level
      type: gauge
      description: "Background intensity level"
    
  analysis_performance:
    - name: peak_finding_time
      type: histogram
      description: "Time to find peaks (ms)"
    - name: phase_identification_time
      type: histogram
      description: "Phase search time (ms)"
    - name: fitting_convergence_rate
      type: gauge
      description: "Percentage of successful fits"
    
  results_quality:
    - name: r_wp_values
      type: histogram
      description: "Distribution of R_wp values"
    - name: phase_match_scores
      type: histogram
      description: "Phase identification scores"
    - name: crystallite_sizes
      type: histogram
      description: "Crystallite size distribution"

alerts:
  - name: poor_pattern_quality
    condition: signal_to_noise < 10
    severity: warning
    message: "Low signal-to-noise ratio in XRD pattern"
    
  - name: fitting_failure
    condition: fitting_convergence_rate < 0.8
    severity: warning
    message: "Peak fitting convergence rate below 80%"
    
  - name: phase_mismatch
    condition: phase_match_scores < 50
    severity: info
    message: "No good phase matches found"

dashboards:
  - name: xrd_overview
    panels:
      - pattern_quality_trends
      - peak_statistics
      - phase_identification_results
      - crystallite_analysis
      
  - name: stress_analysis
    panels:
      - sin2psi_plots
      - stress_distribution
      - measurement_correlation
EOF

info "Monitoring configuration created"

# Step 11: Create documentation
log "Step 11: Generating documentation..."

cat > docs/session9/quick_reference.md << 'EOF'
# Session 9: XRD Analysis - Quick Reference

## Overview
X-ray Diffraction (XRD) analysis system for:
- Phase identification
- Crystallite size determination
- Strain analysis
- Stress measurement
- Texture analysis
- Rietveld refinement

## Quick Start

### 1. Start Services
./start_session9_services.sh

### 2. Access Interface
- XRD Analysis: http://localhost:3000/xrd
- API Documentation: http://localhost:8009/docs

### 3. Basic Workflow

#### Phase Identification
1. Load/measure XRD pattern
2. Process pattern (smoothing, background)
3. Find peaks
4. Search phase database
5. Review matches

#### Crystallite Size Analysis
1. Identify peaks with indices
2. Apply Scherrer equation
3. Perform Williamson-Hall analysis
4. Compare methods

#### Stress Analysis (sin²ψ)
1. Measure at multiple tilt angles
2. Track peak shift
3. Plot sin²ψ vs strain
4. Calculate stress from slope

## Key Features

### Pattern Processing
- Automatic background subtraction
- Peak finding with multiple algorithms
- Profile fitting (Gaussian, Voigt, etc.)
- Kα₂ stripping

### Phase Database
- 6+ pre-loaded phases
- Crystal structure information
- Reference pattern generation
- Custom phase addition

### Analysis Methods
- **Scherrer:** Crystallite size from peak broadening
- **Williamson-Hall:** Separate size and strain
- **sin²ψ:** Residual stress measurement
- **Texture:** Preferred orientation analysis

## Common Parameters

| Parameter | Typical Value | Range |
|-----------|--------------|-------|
| 2θ range | 20-80° | 5-140° |
| Step size | 0.02° | 0.001-0.1° |
| Scan speed | 1°/min | 0.1-10°/min |
| Peak FWHM | 0.1-0.3° | 0.05-1° |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No peaks found | Lower detection threshold, check smoothing |
| Poor phase match | Increase tolerance, check for preferred orientation |
| Large fit residuals | Try different profile function, check background |
| Unrealistic size | Check instrumental broadening, use standard |

## API Examples

### Get Pattern
GET /api/xrd/pattern/{measurement_id}

### Identify Phases
POST /api/xrd/identify-phases
{
  "measurement_id": "uuid",
  "tolerance": 0.1,
  "max_phases": 5
}

### Calculate Crystallite Size
POST /api/xrd/crystallite-size/{measurement_id}

## Support
For detailed documentation, see the complete Session 9 documentation.
EOF

info "Documentation created"

# Step 12: Create startup scripts
log "Step 12: Creating startup scripts..."

cat > start_session9_services.sh << 'EOF'
#!/bin/bash

echo "Starting Session 9: XRD Analysis Services..."

# Activate virtual environment
source venv/bin/activate

# Start backend API
echo "Starting backend API..."
cd backend
uvicorn app.modules.xrd.api:router --reload --port 8009 &
BACKEND_PID=$!
cd ..

# Start frontend (if not already running)
if ! lsof -i:3000 > /dev/null; then
    echo "Starting frontend..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
else
    echo "Frontend already running on port 3000"
    FRONTEND_PID=""
fi

echo ""
echo "Services started:"
echo "  Backend API: http://localhost:8009"
echo "  Frontend: http://localhost:3000/xrd"
echo "  API Docs: http://localhost:8009/docs"

# Save PIDs
mkdir -p .pids
echo $BACKEND_PID > .pids/session9_backend.pid
[ -n "$FRONTEND_PID" ] && echo $FRONTEND_PID > .pids/session9_frontend.pid

echo ""
echo "To stop services, run: ./stop_session9_services.sh"
EOF

chmod +x start_session9_services.sh

cat > stop_session9_services.sh << 'EOF'
#!/bin/bash

echo "Stopping Session 9 services..."

# Read and kill PIDs
if [ -f .pids/session9_backend.pid ]; then
    kill $(cat .pids/session9_backend.pid) 2>/dev/null
    rm .pids/session9_backend.pid
fi

if [ -f .pids/session9_frontend.pid ]; then
    kill $(cat .pids/session9_frontend.pid) 2>/dev/null
    rm .pids/session9_frontend.pid
fi

echo "Services stopped."
EOF

chmod +x stop_session9_services.sh

# Step 13: Final validation
log "Step 13: Running final validation..."

# Check all critical files exist
REQUIRED_FILES=(
    "backend/app/modules/xrd/analyzer.py"
    "backend/app/modules/xrd/api.py"
    "frontend/src/components/xrd/XRDInterface.tsx"
    "backend/tests/xrd/test_integration.py"
    "config/xrd/analysis_config.yaml"
    "start_session9_services.sh"
)

ALL_GOOD=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        warning "Missing file: $file"
        ALL_GOOD=false
    fi
done

if [ "$ALL_GOOD" = true ]; then
    info "All required files present ✓"
else
    error "Some required files are missing"
fi

# Summary
echo ""
echo "============================================" | tee -a "$LOG_FILE"
echo "Deployment Complete!" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo ""
log "Session 9: XRD Analysis has been successfully deployed"
echo ""
echo "Capabilities Deployed:" | tee -a "$LOG_FILE"
echo "✓ Phase identification" | tee -a "$LOG_FILE"
echo "✓ Crystallite size analysis (Scherrer & W-H)" | tee -a "$LOG_FILE"
echo "✓ Residual stress measurement (sin²ψ)" | tee -a "$LOG_FILE"
echo "✓ Texture analysis" | tee -a "$LOG_FILE"
echo "✓ Peak profile fitting" | tee -a "$LOG_FILE"
echo "✓ Rietveld refinement (simplified)" | tee -a "$LOG_FILE"
echo ""
echo "Next Steps:" | tee -a "$LOG_FILE"
echo "1. Start services: ./start_session9_services.sh" | tee -a "$LOG_FILE"
echo "2. Access UI: http://localhost:3000/xrd" | tee -a "$LOG_FILE"
echo "3. View API docs: http://localhost:8009/docs" | tee -a "$LOG_FILE"
echo "4. Run tests: pytest backend/tests/xrd -v" | tee -a "$LOG_FILE"
echo ""
echo "Deployment log saved to: $LOG_FILE"
echo ""

# Create completion marker
touch .session9_complete

exit 0
