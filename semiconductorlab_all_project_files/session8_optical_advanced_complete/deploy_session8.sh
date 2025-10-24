#!/bin/bash

################################################################################
# Session 8: Optical Methods II - Deployment Script
# Ellipsometry, Photoluminescence, and Raman Spectroscopy Implementation
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
SESSION_NAME="session8-optical-advanced"
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
echo "Session 8: Optical Methods II Deployment" | tee -a "$LOG_FILE"
echo "Ellipsometry, PL, and Raman Implementation" | tee -a "$LOG_FILE"
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
if [ ! -f ".session7_complete" ]; then
    warning "Session 7 not found. Some dependencies may be missing."
fi

# Step 2: Create project structure
log "Step 2: Creating project structure..."

# Create directories
DIRS=(
    "backend/app/modules/optical/ellipsometry"
    "backend/app/modules/optical/photoluminescence"
    "backend/app/modules/optical/raman"
    "backend/tests/optical/session8"
    "frontend/src/components/optical/ellipsometry"
    "frontend/src/components/optical/pl"
    "frontend/src/components/optical/raman"
    "data/optical/ellipsometry/models"
    "data/optical/pl/spectra"
    "data/optical/raman/references"
    "docs/session8"
    "logs/optical/session8"
    "config/optical/session8"
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

# Install additional dependencies for Session 8
cat > requirements_session8.txt << EOF
# Core scientific computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

# Optical modeling specific
lmfit>=1.0.3           # Advanced fitting
tmm>=0.1.7            # Transfer matrix method
pyElli>=0.1.0         # Ellipsometry models (optional)
ReFRACtor>=0.1.0      # Refractive index database (optional)

# Signal processing
scikit-learn>=1.0.0
scikit-image>=0.19.0
peakutils>=1.3.3

# Visualization
matplotlib>=3.4.0
plotly>=5.0.0

# Database and API
sqlalchemy>=1.4.0
fastapi>=0.85.0
uvicorn>=0.18.0
redis>=4.3.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.19.0
pytest-cov>=3.0.0

# Units and uncertainties
pint>=0.19.0
uncertainties>=3.1.6
EOF

pip install -r requirements_session8.txt > /dev/null 2>&1

if [ $? -eq 0 ]; then
    info "Python dependencies installed successfully"
else
    warning "Some dependencies may not have installed correctly"
fi

# Step 4: Deploy implementation files
log "Step 4: Deploying implementation files..."

# Copy Python modules
cp session8_complete_implementation.py backend/app/modules/optical/advanced.py
cp test_session8_integration.py backend/tests/optical/session8/test_integration.py

# Copy React components
cp session8_ui_components.tsx frontend/src/components/optical/AdvancedOpticalInterface.tsx

info "Implementation files deployed"

# Step 5: Database setup
log "Step 5: Setting up database tables..."

# Create SQL migration for Session 8
cat > backend/migrations/008_optical_advanced_tables.sql << 'EOF'
-- Session 8: Advanced Optical Methods Tables

-- Ellipsometry measurements table
CREATE TABLE IF NOT EXISTS ellipsometry_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    wavelength_start FLOAT NOT NULL,
    wavelength_end FLOAT NOT NULL,
    num_wavelengths INTEGER NOT NULL,
    angle_of_incidence FLOAT NOT NULL,
    polarizer_angle FLOAT,
    analyzer_angle FLOAT,
    compensator_angle FLOAT,
    instrument_id UUID REFERENCES instruments(id),
    operator_id UUID REFERENCES users(id),
    temperature FLOAT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ellipsometry data
CREATE TABLE IF NOT EXISTS ellipsometry_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES ellipsometry_measurements(id) ON DELETE CASCADE,
    wavelength FLOAT[] NOT NULL,
    psi FLOAT[] NOT NULL,
    delta FLOAT[] NOT NULL,
    depolarization FLOAT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ellipsometry model results
CREATE TABLE IF NOT EXISTS ellipsometry_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES ellipsometry_measurements(id) ON DELETE CASCADE,
    model_type VARCHAR(50) NOT NULL,
    layer_stack JSONB NOT NULL,
    fitted_parameters JSONB,
    mse FLOAT,
    r_squared FLOAT,
    chi_squared FLOAT,
    confidence_intervals JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Photoluminescence measurements
CREATE TABLE IF NOT EXISTS pl_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    excitation_wavelength FLOAT NOT NULL,
    excitation_power FLOAT NOT NULL,
    temperature FLOAT NOT NULL,
    integration_time FLOAT NOT NULL,
    measurement_type VARCHAR(50) DEFAULT 'steady_state',
    grating_density INTEGER,
    slit_width FLOAT,
    instrument_id UUID REFERENCES instruments(id),
    operator_id UUID REFERENCES users(id),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PL spectra data
CREATE TABLE IF NOT EXISTS pl_spectra (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES pl_measurements(id) ON DELETE CASCADE,
    wavelength FLOAT[] NOT NULL,
    intensity FLOAT[] NOT NULL,
    intensity_corrected FLOAT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- PL peaks analysis
CREATE TABLE IF NOT EXISTS pl_peaks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES pl_measurements(id) ON DELETE CASCADE,
    wavelength FLOAT NOT NULL,
    energy FLOAT NOT NULL,
    intensity FLOAT NOT NULL,
    fwhm_nm FLOAT,
    fwhm_meV FLOAT,
    assignment VARCHAR(100),
    area FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Temperature-dependent PL results
CREATE TABLE IF NOT EXISTS pl_temperature_series (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    measurement_ids UUID[] NOT NULL,
    temperatures FLOAT[] NOT NULL,
    peak_energies FLOAT[],
    peak_intensities FLOAT[],
    eg0 FLOAT,  -- Varshni Eg(0)
    alpha FLOAT,  -- Varshni alpha
    beta FLOAT,  -- Varshni beta
    activation_energy FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raman measurements
CREATE TABLE IF NOT EXISTS raman_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    laser_wavelength FLOAT NOT NULL,
    laser_power FLOAT NOT NULL,
    acquisition_time FLOAT NOT NULL,
    accumulations INTEGER DEFAULT 1,
    grating_density INTEGER,
    confocal BOOLEAN DEFAULT FALSE,
    instrument_id UUID REFERENCES instruments(id),
    operator_id UUID REFERENCES users(id),
    x_position FLOAT,
    y_position FLOAT,
    z_position FLOAT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raman spectra data
CREATE TABLE IF NOT EXISTS raman_spectra (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES raman_measurements(id) ON DELETE CASCADE,
    raman_shift FLOAT[] NOT NULL,
    intensity FLOAT[] NOT NULL,
    intensity_normalized FLOAT[],
    baseline FLOAT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raman peaks analysis
CREATE TABLE IF NOT EXISTS raman_peaks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES raman_measurements(id) ON DELETE CASCADE,
    position FLOAT NOT NULL,
    intensity FLOAT NOT NULL,
    fwhm FLOAT,
    area FLOAT,
    material VARCHAR(50),
    mode VARCHAR(100),
    assignment VARCHAR(200),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raman stress/strain analysis
CREATE TABLE IF NOT EXISTS raman_stress_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES raman_measurements(id) ON DELETE CASCADE,
    reference_position FLOAT NOT NULL,
    measured_position FLOAT NOT NULL,
    shift FLOAT NOT NULL,
    strain FLOAT,
    stress_gpa FLOAT,
    stress_type VARCHAR(20),
    material VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raman mapping data
CREATE TABLE IF NOT EXISTS raman_maps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID REFERENCES samples(id),
    measurement_ids UUID[] NOT NULL,
    x_positions FLOAT[] NOT NULL,
    y_positions FLOAT[] NOT NULL,
    map_type VARCHAR(50),  -- intensity, position, fwhm, stress
    map_data FLOAT[][],
    peak_of_interest FLOAT,
    uniformity FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_ellipsometry_measurements_sample ON ellipsometry_measurements(sample_id);
CREATE INDEX idx_pl_measurements_sample ON pl_measurements(sample_id);
CREATE INDEX idx_pl_measurements_temperature ON pl_measurements(temperature);
CREATE INDEX idx_raman_measurements_sample ON raman_measurements(sample_id);
CREATE INDEX idx_raman_measurements_position ON raman_measurements(x_position, y_position);
CREATE INDEX idx_raman_peaks_material ON raman_peaks(material);
CREATE INDEX idx_raman_stress_material ON raman_stress_analysis(material);

-- Add comments
COMMENT ON TABLE ellipsometry_measurements IS 'Spectroscopic ellipsometry measurements';
COMMENT ON TABLE pl_measurements IS 'Photoluminescence spectroscopy measurements';
COMMENT ON TABLE raman_measurements IS 'Raman spectroscopy measurements';
COMMENT ON TABLE ellipsometry_models IS 'Fitted optical models from ellipsometry';
COMMENT ON TABLE pl_temperature_series IS 'Temperature-dependent PL analysis results';
COMMENT ON TABLE raman_stress_analysis IS 'Stress/strain analysis from Raman shifts';
COMMENT ON TABLE raman_maps IS 'Spatial Raman mapping data';
EOF

# Apply migration if PostgreSQL is available
if command -v psql &> /dev/null; then
    psql -U postgres -d semiconductor_lab -f backend/migrations/008_optical_advanced_tables.sql > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        info "Database tables created successfully"
    else
        warning "Could not apply database migrations. Please run manually."
    fi
else
    warning "PostgreSQL not found. Please run migrations manually."
fi

# Step 6: Generate test data and reference materials
log "Step 6: Generating test data and references..."

python3 << 'EOF'
import sys
sys.path.append('.')
from session8_complete_implementation import OpticalTestDataGeneratorII
import json
import numpy as np

# Initialize generator
generator = OpticalTestDataGeneratorII(seed=42)

# Generate ellipsometry reference data
from session8_complete_implementation import LayerStack, DispersionModel

# SiO2 on Si
sio2_stack = LayerStack(
    layers=[{
        'thickness': 100,
        'model': DispersionModel.CAUCHY,
        'params': {'A': 1.46, 'B': 0.00354, 'C': 0, 'k': 0}
    }],
    substrate={'n': 3.85, 'k': 0.02}
)

ell_data = generator.generate_ellipsometry_data(sio2_stack)
data = {
    'wavelength': ell_data.wavelength.tolist(),
    'psi': ell_data.psi.tolist(),
    'delta': ell_data.delta.tolist(),
    'angle': ell_data.angle_of_incidence,
    'description': 'SiO2 (100nm) on Si substrate'
}

with open('data/optical/ellipsometry/models/sio2_on_si.json', 'w') as f:
    json.dump(data, f, indent=2)

# Generate PL reference data
materials = ['GaAs', 'GaN', 'InP']
for material in materials:
    pl_spectrum = generator.generate_pl_spectrum(material, temperature=10)
    data = {
        'material': material,
        'wavelength': pl_spectrum.wavelength.tolist(),
        'intensity': pl_spectrum.intensity.tolist(),
        'temperature': pl_spectrum.temperature,
        'excitation': pl_spectrum.excitation_wavelength
    }
    
    with open(f'data/optical/pl/spectra/{material.lower()}_10K.json', 'w') as f:
        json.dump(data, f, indent=2)

# Generate Raman reference data
materials = ['Si', 'GaAs', 'Graphene']
for material in materials:
    raman_spectrum = generator.generate_raman_spectrum(material)
    data = {
        'material': material,
        'raman_shift': raman_spectrum.raman_shift.tolist(),
        'intensity': raman_spectrum.intensity.tolist(),
        'laser': raman_spectrum.laser_wavelength
    }
    
    with open(f'data/optical/raman/references/{material.lower()}_raman.json', 'w') as f:
        json.dump(data, f, indent=2)

print("Reference data generated successfully")
EOF

if [ $? -eq 0 ]; then
    info "Test data and references generated successfully"
else
    warning "Test data generation encountered issues"
fi

# Step 7: Configure optical models database
log "Step 7: Configuring optical models database..."

cat > config/optical/session8/dispersion_models.yaml << 'EOF'
# Session 8: Dispersion Models Configuration

materials:
  # Semiconductors
  Si:
    model: sellmeier
    parameters:
      B1: 10.6684293
      C1: 0.301516485
      B2: 0.003043475
      C2: 1.13475115
      B3: 1.54133408
      C3: 1104.0
    reference: "Aspnes and Studna, PRB 27, 985 (1983)"
    
  GaAs:
    model: tauc_lorentz
    parameters:
      A: 150
      E0: 4.5
      C: 2.0
      Eg: 1.42
      eps_inf: 10.9
    reference: "Jellison and Modine, APL 69, 371 (1996)"
    
  SiO2:
    model: cauchy
    parameters:
      A: 1.4584
      B: 0.00354
      C: 0.00001
      k: 0
    wavelength_range: [200, 1200]
    
  Si3N4:
    model: cauchy
    parameters:
      A: 2.0029
      B: 0.00757
      C: 0.00002
      k: 0
    wavelength_range: [300, 1500]
    
  Al2O3:
    model: cauchy
    parameters:
      A: 1.7659
      B: 0.00565
      C: 0.00001
      k: 0
    wavelength_range: [200, 2000]

# PL emission data
pl_materials:
  GaAs:
    bandgap_300K: 1.42
    bandgap_0K: 1.519
    varshni_alpha: 5.405e-4
    varshni_beta: 204
    exciton_binding: 4.2  # meV
    
  GaN:
    bandgap_300K: 3.4
    bandgap_0K: 3.503
    varshni_alpha: 7.7e-4
    varshni_beta: 600
    exciton_binding: 25  # meV
    
  InP:
    bandgap_300K: 1.35
    bandgap_0K: 1.421
    varshni_alpha: 4.9e-4
    varshni_beta: 327
    exciton_binding: 4.8  # meV

# Raman modes database
raman_modes:
  Si:
    - mode: "TO/LO"
      position: 520.5
      temperature_coeff: -0.021  # cm⁻¹/K
      stress_coeff: -1.8  # cm⁻¹/GPa
      
  GaAs:
    - mode: "TO(Γ)"
      position: 268
      temperature_coeff: -0.016
      stress_coeff: -1.5
    - mode: "LO(Γ)"
      position: 292
      temperature_coeff: -0.013
      stress_coeff: -1.2
      
  Graphene:
    - mode: "G"
      position: 1580
      fwhm_pristine: 15
    - mode: "2D"
      position: 2700
      fwhm_pristine: 30
    - mode: "D"
      position: 1350
      defect_activated: true
EOF

info "Optical models database configured"

# Step 8: Create API endpoints
log "Step 8: Setting up API endpoints..."

cat > backend/app/modules/optical/session8_api.py << 'EOF'
"""
Session 8: Advanced Optical Methods API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID
import numpy as np

from .advanced import (
    EllipsometryAnalyzer, PhotoluminescenceAnalyzer, RamanAnalyzer,
    EllipsometryData, LayerStack, PLSpectrum, RamanSpectrum,
    DispersionModel, PLMeasurementType, RamanMode
)

router = APIRouter(prefix="/api/optical/advanced", tags=["optical-advanced"])

# Pydantic models
class EllipsometryMeasurementRequest(BaseModel):
    sample_id: UUID
    wavelength_start: float = Field(gt=0)
    wavelength_end: float = Field(gt=0)
    angle_of_incidence: float = Field(ge=0, le=90)
    
class LayerModelRequest(BaseModel):
    layers: List[Dict[str, Any]]
    substrate: Dict[str, Any]
    fit_parameters: List[str]
    
class PLMeasurementRequest(BaseModel):
    sample_id: UUID
    excitation_wavelength: float
    excitation_power: float
    temperature: float
    integration_time: float = 1.0
    
class RamanMeasurementRequest(BaseModel):
    sample_id: UUID
    laser_wavelength: float
    laser_power: float
    acquisition_time: float = 10.0
    
# Ellipsometry endpoints
@router.post("/ellipsometry/measure")
async def measure_ellipsometry(request: EllipsometryMeasurementRequest):
    """Start ellipsometry measurement"""
    try:
        # Would interface with instrument
        return {
            "status": "measurement_started",
            "measurement_id": "generated_uuid",
            "estimated_time": 120  # seconds
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ellipsometry/fit-model")
async def fit_ellipsometry_model(
    measurement_id: UUID,
    model: LayerModelRequest,
    background_tasks: BackgroundTasks
):
    """Fit optical model to ellipsometry data"""
    try:
        # Would run fitting in background
        background_tasks.add_task(run_ellipsometry_fitting, measurement_id, model)
        return {
            "status": "fitting_started",
            "job_id": "generated_uuid"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PL endpoints
@router.post("/pl/measure")
async def measure_pl(request: PLMeasurementRequest):
    """Start PL measurement"""
    try:
        return {
            "status": "measurement_started",
            "measurement_id": "generated_uuid",
            "temperature_stable": request.temperature < 100
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pl/temperature-series")
async def measure_pl_temperature_series(
    sample_id: UUID,
    temperatures: List[float],
    excitation_wavelength: float,
    background_tasks: BackgroundTasks
):
    """Run temperature-dependent PL series"""
    try:
        background_tasks.add_task(
            run_temperature_series, 
            sample_id, 
            temperatures, 
            excitation_wavelength
        )
        return {
            "status": "series_started",
            "num_temperatures": len(temperatures),
            "estimated_time": len(temperatures) * 60
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Raman endpoints
@router.post("/raman/measure")
async def measure_raman(request: RamanMeasurementRequest):
    """Start Raman measurement"""
    try:
        return {
            "status": "measurement_started",
            "measurement_id": "generated_uuid"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/raman/stress-analysis")
async def analyze_raman_stress(
    measurement_id: UUID,
    material: str,
    reference_position: float
):
    """Analyze stress from Raman shift"""
    try:
        # Would fetch data and analyze
        analyzer = RamanAnalyzer()
        # Mock result
        return {
            "stress_gpa": 0.5,
            "strain": 0.001,
            "type": "compressive",
            "shift": 2.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/raman/map")
async def start_raman_mapping(
    sample_id: UUID,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    step_size: float,
    background_tasks: BackgroundTasks
):
    """Start Raman mapping"""
    try:
        # Calculate number of points
        nx = int((x_range[1] - x_range[0]) / step_size) + 1
        ny = int((y_range[1] - y_range[0]) / step_size) + 1
        
        background_tasks.add_task(
            run_raman_mapping,
            sample_id,
            x_range,
            y_range,
            step_size
        )
        
        return {
            "status": "mapping_started",
            "num_points": nx * ny,
            "estimated_time": nx * ny * 10  # seconds
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "module": "optical-advanced",
        "session": 8,
        "methods": ["ellipsometry", "photoluminescence", "raman"]
    }

# Background task functions (would be implemented with actual logic)
async def run_ellipsometry_fitting(measurement_id, model):
    pass

async def run_temperature_series(sample_id, temperatures, excitation):
    pass

async def run_raman_mapping(sample_id, x_range, y_range, step_size):
    pass
EOF

info "API endpoints configured"

# Step 9: Run tests
log "Step 9: Running integration tests..."

python -m pytest backend/tests/optical/session8/test_integration.py -v --tb=short -k "not performance" > test_results.log 2>&1

if [ $? -eq 0 ]; then
    info "All tests passed ✓"
else
    warning "Some tests failed. Check test_results.log for details"
fi

# Step 10: Build Docker containers
log "Step 10: Building Docker containers..."

if command -v docker &> /dev/null; then
    cat > Dockerfile.session8 << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_session8.txt .
RUN pip install --no-cache-dir -r requirements_session8.txt

# Copy application code
COPY backend/app/modules/optical /app/optical
COPY config/optical/session8 /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODULE_NAME=optical-advanced

# Run the application
CMD ["uvicorn", "optical.session8_api:router", "--host", "0.0.0.0", "--port", "8008"]
EOF

    docker build -f Dockerfile.session8 -t ${PROJECT_NAME}/optical-advanced:session8 . > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        info "Docker image built successfully"
    else
        warning "Docker build encountered issues"
    fi
else
    warning "Docker not available. Skipping container build."
fi

# Step 11: Create monitoring dashboards
log "Step 11: Setting up monitoring..."

cat > config/optical/session8/monitoring.yaml << 'EOF'
# Session 8 Monitoring Configuration

metrics:
  ellipsometry:
    - name: fit_convergence_rate
      type: gauge
      description: "Percentage of successful model fits"
    - name: average_mse
      type: histogram
      description: "Mean squared error distribution"
    - name: measurement_duration
      type: histogram
      description: "Time taken for full spectrum measurement"
      
  photoluminescence:
    - name: peak_detection_rate
      type: gauge
      description: "Success rate of automatic peak detection"
    - name: temperature_stability
      type: gauge
      description: "Temperature control stability (mK)"
    - name: quantum_yield
      type: histogram
      description: "Calculated quantum yield values"
      
  raman:
    - name: peak_shift_variance
      type: gauge
      description: "Variance in peak position (cm⁻¹)"
    - name: stress_measurements
      type: counter
      description: "Number of stress measurements performed"
    - name: mapping_completion_time
      type: histogram
      description: "Time to complete spatial mapping"

alerts:
  - name: ellipsometry_fit_failure
    condition: fit_convergence_rate < 0.8
    severity: warning
    message: "Ellipsometry fitting success rate below 80%"
    
  - name: pl_temperature_drift
    condition: temperature_stability > 100  # mK
    severity: warning
    message: "PL temperature stability exceeds 100 mK"
    
  - name: raman_peak_shift
    condition: peak_shift_variance > 2  # cm⁻¹
    severity: warning
    message: "Raman peak position variance exceeds 2 cm⁻¹"
EOF

info "Monitoring configuration created"

# Step 12: Create startup scripts
log "Step 12: Creating startup scripts..."

cat > start_session8_services.sh << 'EOF'
#!/bin/bash

echo "Starting Session 8: Advanced Optical Services..."

# Activate virtual environment
source venv/bin/activate

# Start backend API
echo "Starting backend API..."
cd backend
uvicorn app.modules.optical.session8_api:router --reload --port 8008 &
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
echo "  Backend API: http://localhost:8008"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8008/docs"

# Save PIDs
mkdir -p .pids
echo $BACKEND_PID > .pids/session8_backend.pid
[ -n "$FRONTEND_PID" ] && echo $FRONTEND_PID > .pids/session8_frontend.pid

echo ""
echo "To stop services, run: ./stop_session8_services.sh"
EOF

chmod +x start_session8_services.sh

cat > stop_session8_services.sh << 'EOF'
#!/bin/bash

echo "Stopping Session 8 services..."

# Read and kill PIDs
if [ -f .pids/session8_backend.pid ]; then
    kill $(cat .pids/session8_backend.pid) 2>/dev/null
    rm .pids/session8_backend.pid
fi

if [ -f .pids/session8_frontend.pid ]; then
    kill $(cat .pids/session8_frontend.pid) 2>/dev/null
    rm .pids/session8_frontend.pid
fi

echo "Services stopped."
EOF

chmod +x stop_session8_services.sh

# Step 13: Generate training materials
log "Step 13: Creating training materials..."

cat > docs/session8/quick_start_guide.md << 'EOF'
# Session 8: Advanced Optical Methods - Quick Start Guide

## Overview
This session implements three advanced optical characterization techniques:
- **Ellipsometry**: Multi-layer thin film analysis
- **Photoluminescence (PL)**: Emission spectroscopy
- **Raman Spectroscopy**: Vibrational analysis and stress measurement

## Quick Start

### 1. Start Services
```bash
./start_session8_services.sh
```

### 2. Access Interfaces
- Ellipsometry: http://localhost:3000/optical/ellipsometry
- PL: http://localhost:3000/optical/pl
- Raman: http://localhost:3000/optical/raman

### 3. Run Demo Measurements

#### Ellipsometry Demo
1. Click "Load Demo Data" in Setup tab
2. Define layer stack in Model tab
3. Click "Fit Model" in Analysis tab
4. View results in Results tab

#### PL Demo
1. Set temperature to 10K
2. Click "Start Measurement"
3. Run "Temperature Series" for Varshni analysis
4. View peak analysis in Results

#### Raman Demo
1. Load Si reference spectrum
2. View peak identification
3. Check stress analysis (shift from 520.5 cm⁻¹)
4. Review crystallinity assessment

## Key Features

### Ellipsometry
- Transfer Matrix Method modeling
- Multiple dispersion models (Cauchy, Sellmeier, Tauc-Lorentz)
- Multi-layer stack fitting
- MSE minimization with constraints

### Photoluminescence
- Temperature-dependent measurements
- Varshni equation fitting
- Multi-peak deconvolution
- Quantum yield calculation

### Raman
- Automatic peak identification
- Stress/strain calculation
- Crystallinity analysis
- Spatial mapping capability

## Common Workflows

### Film Thickness Measurement (Ellipsometry)
1. Measure Ψ and Δ vs wavelength
2. Create layer model with initial guess
3. Fit thickness as free parameter
4. Validate with MSE < 5

### Bandgap Determination (PL)
1. Cool sample to low temperature
2. Measure PL spectrum
3. Identify band-edge emission
4. Extract bandgap from peak position

### Stress Analysis (Raman)
1. Measure Raman spectrum
2. Identify material peaks (e.g., Si at 520.5 cm⁻¹)
3. Calculate shift from reference
4. Convert to stress using material constants

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Poor ellipsometry fit | Check initial parameters, add roughness layer |
| No PL signal | Increase laser power, check alignment |
| Broad Raman peaks | Reduce laser power, check for damage |
| Temperature instability | Wait for equilibration, check cryostat |

## Support
For detailed documentation, see the complete Session 8 documentation.
EOF

info "Training materials created"

# Step 14: Final validation
log "Step 14: Running final validation..."

# Check all critical files exist
REQUIRED_FILES=(
    "backend/app/modules/optical/advanced.py"
    "backend/app/modules/optical/session8_api.py"
    "frontend/src/components/optical/AdvancedOpticalInterface.tsx"
    "backend/tests/optical/session8/test_integration.py"
    "config/optical/session8/dispersion_models.yaml"
    "start_session8_services.sh"
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
log "Session 8: Advanced Optical Methods has been successfully deployed"
echo ""
echo "Capabilities Deployed:" | tee -a "$LOG_FILE"
echo "✓ Spectroscopic Ellipsometry" | tee -a "$LOG_FILE"
echo "✓ Photoluminescence Spectroscopy" | tee -a "$LOG_FILE"
echo "✓ Raman Spectroscopy" | tee -a "$LOG_FILE"
echo "✓ Temperature-dependent measurements" | tee -a "$LOG_FILE"
echo "✓ Stress/strain analysis" | tee -a "$LOG_FILE"
echo "✓ Multi-layer optical modeling" | tee -a "$LOG_FILE"
echo ""
echo "Next Steps:" | tee -a "$LOG_FILE"
echo "1. Start services: ./start_session8_services.sh" | tee -a "$LOG_FILE"
echo "2. Access UI: http://localhost:3000/optical" | tee -a "$LOG_FILE"
echo "3. View API docs: http://localhost:8008/docs" | tee -a "$LOG_FILE"
echo "4. Run tests: pytest backend/tests/optical/session8 -v" | tee -a "$LOG_FILE"
echo ""
echo "Deployment log saved to: $LOG_FILE"
echo ""

# Create completion marker
touch .session8_complete

exit 0
