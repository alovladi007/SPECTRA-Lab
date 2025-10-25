#!/bin/bash

################################################################################
# Session 12: Chemical II (SIMS/RBS/NAA/Etch) - Deployment Script
################################################################################
#
# Deploys complete chemical and bulk analysis infrastructure:
# - Database schema (SIMS, RBS, NAA, Etch tables)
# - FastAPI service configuration
# - Calibration data loading
# - Test data generation
# - Service orchestration
#
# Author: Semiconductor Lab Platform Team
# Version: 1.0.0
# Date: October 2024
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="chemical-bulk-analysis"
SERVICE_PORT=8012
DB_NAME="semiconductor_lab"
DB_USER="lab_user"
SCHEMA_VERSION="1.0.0"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Session 12: Chemical & Bulk Analysis Deployment      ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

################################################################################
# Pre-flight Checks
################################################################################

check_dependencies() {
    echo -e "${YELLOW}→${NC} Checking dependencies..."
    
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check PostgreSQL
    if ! command -v psql &> /dev/null; then
        missing_deps+=("postgresql-client")
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("python3-pip")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${RED}✗${NC} Missing dependencies: ${missing_deps[*]}"
        echo "  Install with: sudo apt-get install ${missing_deps[*]}"
        exit 1
    fi
    
    echo -e "${GREEN}✓${NC} All dependencies satisfied"
}

check_python_packages() {
    echo -e "${YELLOW}→${NC} Checking Python packages..."
    
    local required_packages=(
        "fastapi"
        "uvicorn"
        "numpy"
        "scipy"
        "pandas"
        "pydantic"
        "psycopg2-binary"
        "sqlalchemy"
    )
    
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import ${package//-/_}" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        echo -e "${YELLOW}!${NC} Missing packages: ${missing_packages[*]}"
        echo -e "${YELLOW}→${NC} Installing..."
        pip3 install "${missing_packages[@]}" --user
    fi
    
    echo -e "${GREEN}✓${NC} All Python packages available"
}

################################################################################
# Database Setup
################################################################################

create_database_schema() {
    echo -e "${YELLOW}→${NC} Creating database schema..."
    
    # SQL schema for Session 12
    cat > /tmp/session12_schema.sql << 'EOF'
-- ============================================================================
-- Session 12: Chemical & Bulk Analysis - Database Schema
-- ============================================================================

-- SIMS (Secondary Ion Mass Spectrometry)
-- ============================================================================

-- SIMS measurements
CREATE TABLE IF NOT EXISTS sims_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID NOT NULL,
    run_id UUID NOT NULL,
    element VARCHAR(10) NOT NULL,
    matrix VARCHAR(20) NOT NULL,
    isotope INTEGER,
    mode VARCHAR(20) DEFAULT 'dynamic',
    primary_ion VARCHAR(20),
    primary_energy FLOAT,
    primary_current FLOAT,
    raster_size FLOAT,
    analysis_area FLOAT,
    sputter_rate FLOAT,
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operator VARCHAR(100),
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SIMS depth profiles
CREATE TABLE IF NOT EXISTS sims_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES sims_measurements(id) ON DELETE CASCADE,
    time FLOAT[],
    depth FLOAT[],
    counts FLOAT[],
    concentration FLOAT[],
    calibration_id UUID,
    quantification_method VARCHAR(50),
    total_dose FLOAT,
    detection_limit FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- SIMS calibrations
CREATE TABLE IF NOT EXISTS sims_calibrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    element VARCHAR(10) NOT NULL,
    matrix VARCHAR(20) NOT NULL,
    rsf FLOAT NOT NULL,
    rsf_uncertainty FLOAT,
    reference_concentration FLOAT,
    sputter_rate FLOAT,
    calibration_date DATE,
    standard_name VARCHAR(100),
    certificate_number VARCHAR(100),
    expiry_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(element, matrix, calibration_date)
);

-- SIMS interfaces
CREATE TABLE IF NOT EXISTS sims_interfaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES sims_profiles(id) ON DELETE CASCADE,
    depth FLOAT NOT NULL,
    width FLOAT,
    gradient FLOAT,
    concentration FLOAT,
    interface_type VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- RBS (Rutherford Backscattering Spectrometry)
-- ============================================================================

-- RBS measurements
CREATE TABLE IF NOT EXISTS rbs_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID NOT NULL,
    run_id UUID NOT NULL,
    incident_energy FLOAT NOT NULL,
    scattering_angle FLOAT NOT NULL,
    detector_solid_angle FLOAT,
    incident_charge FLOAT,
    projectile VARCHAR(10) DEFAULT 'He',
    geometry VARCHAR(20),
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operator VARCHAR(100),
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- RBS spectra
CREATE TABLE IF NOT EXISTS rbs_spectra (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES rbs_measurements(id) ON DELETE CASCADE,
    energy FLOAT[],
    channel INTEGER[],
    counts FLOAT[],
    energy_resolution FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- RBS fitted layers
CREATE TABLE IF NOT EXISTS rbs_layers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spectrum_id UUID REFERENCES rbs_spectra(id) ON DELETE CASCADE,
    layer_number INTEGER NOT NULL,
    element VARCHAR(10) NOT NULL,
    atomic_fraction FLOAT NOT NULL,
    thickness FLOAT NOT NULL,
    thickness_nm FLOAT,
    density FLOAT,
    is_substrate BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- RBS fitting results
CREATE TABLE IF NOT EXISTS rbs_fits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spectrum_id UUID REFERENCES rbs_spectra(id) ON DELETE CASCADE,
    simulated_spectrum FLOAT[],
    chi_squared FLOAT,
    r_factor FLOAT,
    fit_range_start INTEGER,
    fit_range_end INTEGER,
    fix_composition BOOLEAN,
    convergence BOOLEAN,
    iterations INTEGER,
    fit_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- NAA (Neutron Activation Analysis)
-- ============================================================================

-- NAA measurements
CREATE TABLE IF NOT EXISTS naa_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID NOT NULL,
    run_id UUID NOT NULL,
    element VARCHAR(10) NOT NULL,
    isotope VARCHAR(20) NOT NULL,
    gamma_energy FLOAT,
    irradiation_time FLOAT,
    cooling_time FLOAT,
    measurement_time FLOAT,
    neutron_flux FLOAT,
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operator VARCHAR(100),
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NAA decay curves
CREATE TABLE IF NOT EXISTS naa_decay_curves (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES naa_measurements(id) ON DELETE CASCADE,
    time FLOAT[],
    counts FLOAT[],
    live_time FLOAT[],
    background FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NAA quantification results
CREATE TABLE IF NOT EXISTS naa_quantifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES naa_measurements(id) ON DELETE CASCADE,
    standard_id UUID,
    concentration FLOAT NOT NULL,
    concentration_units VARCHAR(20) DEFAULT 'μg/g',
    uncertainty FLOAT,
    detection_limit FLOAT,
    activity FLOAT,
    decay_constant FLOAT,
    half_life FLOAT,
    method VARCHAR(50) DEFAULT 'comparator',
    chi_squared FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NAA nuclear data library
CREATE TABLE IF NOT EXISTS naa_nuclear_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    element VARCHAR(10) NOT NULL,
    isotope VARCHAR(20) NOT NULL,
    half_life FLOAT NOT NULL,
    gamma_energy FLOAT NOT NULL,
    gamma_intensity FLOAT,
    thermal_cross_section FLOAT,
    activation_product VARCHAR(20),
    notes TEXT,
    reference VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(isotope, gamma_energy)
);


-- Chemical Etch
-- ============================================================================

-- Etch measurements
CREATE TABLE IF NOT EXISTS etch_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID NOT NULL,
    run_id UUID NOT NULL,
    chemistry VARCHAR(50) NOT NULL,
    temperature FLOAT,
    concentration FLOAT,
    concentration_units VARCHAR(20),
    etch_time FLOAT,
    agitation VARCHAR(50),
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operator VARCHAR(100),
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Etch profiles
CREATE TABLE IF NOT EXISTS etch_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES etch_measurements(id) ON DELETE CASCADE,
    pattern_density FLOAT[],
    etch_rate FLOAT[],
    position_x FLOAT[],
    position_y FLOAT[],
    measurement_type VARCHAR(50) DEFAULT 'loading_effect',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Etch loading effects
CREATE TABLE IF NOT EXISTS etch_loading_effects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES etch_profiles(id) ON DELETE CASCADE,
    nominal_rate FLOAT NOT NULL,
    max_reduction FLOAT,
    critical_density FLOAT,
    model_type VARCHAR(20) NOT NULL,
    r_squared FLOAT,
    coefficients JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Etch uniformity metrics
CREATE TABLE IF NOT EXISTS etch_uniformity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES etch_profiles(id) ON DELETE CASCADE,
    mean_rate FLOAT NOT NULL,
    std_rate FLOAT,
    uniformity_1sigma FLOAT,
    uniformity_3sigma FLOAT,
    uniformity_range FLOAT,
    min_rate FLOAT,
    max_rate FLOAT,
    cv_percent FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Indexes for Performance
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_sims_measurements_sample ON sims_measurements(sample_id);
CREATE INDEX IF NOT EXISTS idx_sims_measurements_element ON sims_measurements(element);
CREATE INDEX IF NOT EXISTS idx_sims_profiles_measurement ON sims_profiles(measurement_id);
CREATE INDEX IF NOT EXISTS idx_sims_calibrations_element_matrix ON sims_calibrations(element, matrix);

CREATE INDEX IF NOT EXISTS idx_rbs_measurements_sample ON rbs_measurements(sample_id);
CREATE INDEX IF NOT EXISTS idx_rbs_spectra_measurement ON rbs_spectra(measurement_id);
CREATE INDEX IF NOT EXISTS idx_rbs_layers_spectrum ON rbs_layers(spectrum_id);

CREATE INDEX IF NOT EXISTS idx_naa_measurements_sample ON naa_measurements(sample_id);
CREATE INDEX IF NOT EXISTS idx_naa_measurements_element ON naa_measurements(element);
CREATE INDEX IF NOT EXISTS idx_naa_decay_curves_measurement ON naa_decay_curves(measurement_id);

CREATE INDEX IF NOT EXISTS idx_etch_measurements_sample ON etch_measurements(sample_id);
CREATE INDEX IF NOT EXISTS idx_etch_profiles_measurement ON etch_profiles(measurement_id);


-- Schema Version
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_versions (
    module VARCHAR(50) PRIMARY KEY,
    version VARCHAR(20) NOT NULL,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO schema_versions (module, version) 
VALUES ('session12_chemical_bulk', '1.0.0')
ON CONFLICT (module) DO UPDATE SET 
    version = EXCLUDED.version,
    deployed_at = CURRENT_TIMESTAMP;

EOF

    # Execute schema creation
    psql -h localhost -U $DB_USER -d $DB_NAME -f /tmp/session12_schema.sql
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Database schema created successfully"
    else
        echo -e "${RED}✗${NC} Failed to create database schema"
        exit 1
    fi
    
    rm /tmp/session12_schema.sql
}

load_nuclear_data() {
    echo -e "${YELLOW}→${NC} Loading NAA nuclear data library..."
    
    cat > /tmp/load_nuclear_data.sql << 'EOF'
-- Common isotopes for NAA
INSERT INTO naa_nuclear_data (element, isotope, half_life, gamma_energy, gamma_intensity, thermal_cross_section, activation_product) VALUES
    ('Na', 'Na-24', 53996.0, 1368.6, 1.0, 0.53, 'Na-24'),
    ('Mn', 'Mn-56', 9287.0, 846.8, 0.989, 13.3, 'Mn-56'),
    ('Cu', 'Cu-64', 45720.0, 1345.8, 0.005, 4.5, 'Cu-64'),
    ('As', 'As-76', 95040.0, 559.1, 0.45, 4.5, 'As-76'),
    ('Br', 'Br-82', 126230.0, 776.5, 0.835, 6.8, 'Br-82'),
    ('Au', 'Au-198', 232992.0, 411.8, 0.955, 98.65, 'Au-198')
ON CONFLICT (isotope, gamma_energy) DO NOTHING;
EOF

    psql -h localhost -U $DB_USER -d $DB_NAME -f /tmp/load_nuclear_data.sql
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Nuclear data loaded"
    else
        echo -e "${YELLOW}!${NC} Nuclear data loading had issues (may already exist)"
    fi
    
    rm /tmp/load_nuclear_data.sql
}

load_default_calibrations() {
    echo -e "${YELLOW}→${NC} Loading default SIMS calibrations..."
    
    cat > /tmp/load_calibrations.sql << 'EOF'
-- Default RSF values for common dopants in Si
INSERT INTO sims_calibrations (element, matrix, rsf, rsf_uncertainty, sputter_rate, standard_name) VALUES
    ('B', 'Si', 1.8e21, 2.7e20, 1.0, 'default'),
    ('P', 'Si', 3.0e21, 4.5e20, 1.0, 'default'),
    ('As', 'Si', 2.5e21, 3.75e20, 1.0, 'default'),
    ('Sb', 'Si', 2.0e21, 3.0e20, 1.0, 'default'),
    ('In', 'Si', 1.5e21, 2.25e20, 1.0, 'default'),
    ('Ga', 'Si', 1.7e21, 2.55e20, 1.0, 'default'),
    ('Al', 'Si', 2.2e21, 3.3e20, 1.0, 'default'),
    ('N', 'Si', 5.0e21, 7.5e20, 1.0, 'default'),
    ('O', 'Si', 1.0e22, 1.5e21, 1.0, 'default'),
    ('C', 'Si', 3.5e21, 5.25e20, 1.0, 'default')
ON CONFLICT (element, matrix, calibration_date) DO NOTHING;
EOF

    psql -h localhost -U $DB_USER -d $DB_NAME -f /tmp/load_calibrations.sql
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Default calibrations loaded"
    else
        echo -e "${YELLOW}!${NC} Calibration loading had issues"
    fi
    
    rm /tmp/load_calibrations.sql
}

################################################################################
# Service Configuration
################################################################################

create_service_config() {
    echo -e "${YELLOW}→${NC} Creating service configuration..."
    
    # Create config directory
    mkdir -p config/session12
    
    # Analysis parameters
    cat > config/session12/analysis_config.yaml << 'EOF'
# Session 12: Chemical & Bulk Analysis Configuration

sims:
  default_sputter_rate: 1.0  # nm/s
  default_method: "RSF"
  detection_limit_sigma: 3
  interface_threshold: 0.5
  smoothing_window: 5

rbs:
  default_projectile: "He"
  default_energy: 2000.0  # keV
  default_angle: 170.0  # degrees
  detector_resolution: 15.0  # keV FWHM
  fit_tolerance: 1e-6
  max_iterations: 1000

naa:
  default_method: "comparator"
  background_region: [0, 5]  # Last 5 points
  fit_method: "fixed_lambda"  # or "free_lambda"
  detection_limit_sigma: 3

etch:
  default_model: "linear"
  models_available: ["linear", "exponential", "power"]
  fit_tolerance: 1e-6
  uniformity_sigma_levels: [1, 3]

general:
  max_data_points: 10000
  cache_results: true
  result_ttl: 3600  # seconds
EOF

    # Monitoring config
    cat > config/session12/monitoring.yaml << 'EOF'
# Monitoring Configuration

metrics:
  - name: "sims_analysis_time"
    type: "histogram"
    unit: "seconds"
  
  - name: "rbs_fit_iterations"
    type: "gauge"
    unit: "count"
  
  - name: "naa_detection_limit"
    type: "gauge"
    unit: "μg/g"
  
  - name: "etch_uniformity"
    type: "gauge"
    unit: "percent"

alerts:
  - name: "poor_rbs_fit"
    condition: "r_factor > 0.3"
    severity: "warning"
  
  - name: "low_naa_activity"
    condition: "activity < detection_limit * 10"
    severity: "warning"
  
  - name: "high_etch_non_uniformity"
    condition: "cv_percent > 10"
    severity: "warning"
EOF

    echo -e "${GREEN}✓${NC} Service configuration created"
}

create_start_script() {
    echo -e "${YELLOW}→${NC} Creating service start script..."
    
    cat > start_session12_services.sh << 'EOF'
#!/bin/bash

echo "Starting Session 12 services..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export SESSION12_PORT=8012
export SESSION12_CONFIG_DIR="./config/session12"

# Start FastAPI service
echo "Starting Chemical & Bulk Analysis API on port $SESSION12_PORT..."
python3 -m uvicorn session12_chemical_bulk_complete_implementation:app \
    --host 0.0.0.0 \
    --port $SESSION12_PORT \
    --reload \
    --log-level info &

SERVICE_PID=$!
echo "Service started with PID: $SERVICE_PID"
echo $SERVICE_PID > /tmp/session12_service.pid

echo "Session 12 services running!"
echo "API: http://localhost:$SESSION12_PORT"
echo "Docs: http://localhost:$SESSION12_PORT/docs"
EOF

    chmod +x start_session12_services.sh
    echo -e "${GREEN}✓${NC} Start script created"
}

create_stop_script() {
    echo -e "${YELLOW}→${NC} Creating service stop script..."
    
    cat > stop_session12_services.sh << 'EOF'
#!/bin/bash

echo "Stopping Session 12 services..."

if [ -f /tmp/session12_service.pid ]; then
    PID=$(cat /tmp/session12_service.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "Stopped service (PID: $PID)"
    else
        echo "Service not running"
    fi
    rm /tmp/session12_service.pid
else
    echo "No PID file found"
fi

echo "Session 12 services stopped"
EOF

    chmod +x stop_session12_services.sh
    echo -e "${GREEN}✓${NC} Stop script created"
}

################################################################################
# Test Data Generation
################################################################################

generate_test_data() {
    echo -e "${YELLOW}→${NC} Generating test data..."
    
    python3 << 'EOF'
import numpy as np
import json
from session12_chemical_bulk_complete_implementation import ChemicalBulkSimulator

simulator = ChemicalBulkSimulator()

# Generate test datasets
test_data = {}

# SIMS profiles
print("Generating SIMS test data...")
sims_b = simulator.simulate_sims_profile(element="B", peak_depth=100, dose=1e15)
sims_p = simulator.simulate_sims_profile(element="P", peak_depth=80, dose=5e14)

test_data['sims'] = {
    'boron_implant': {
        'depth': sims_b.depth.tolist(),
        'concentration': sims_b.concentration.tolist(),
        'element': sims_b.element
    },
    'phosphorus_implant': {
        'depth': sims_p.depth.tolist(),
        'concentration': sims_p.concentration.tolist(),
        'element': sims_p.element
    }
}

# RBS spectra
print("Generating RBS test data...")
rbs_hfo2 = simulator.simulate_rbs_spectrum([("Hf", 0.5, 20), ("O", 0.5, 20)])
rbs_sio2 = simulator.simulate_rbs_spectrum([("Si", 0.33, 15), ("O", 0.67, 15)])

test_data['rbs'] = {
    'hfo2_film': {
        'energy': rbs_hfo2.energy.tolist(),
        'counts': rbs_hfo2.counts.tolist()
    },
    'sio2_film': {
        'energy': rbs_sio2.energy.tolist(),
        'counts': rbs_sio2.counts.tolist()
    }
}

# NAA decay curves
print("Generating NAA test data...")
naa_au = simulator.simulate_naa_decay(element="Au", initial_activity=10000)
naa_na = simulator.simulate_naa_decay(element="Na", initial_activity=20000)

test_data['naa'] = {
    'gold_analysis': {
        'time': naa_au.time.tolist(),
        'counts': naa_au.counts.tolist(),
        'element': naa_au.element
    },
    'sodium_analysis': {
        'time': naa_na.time.tolist(),
        'counts': naa_na.counts.tolist(),
        'element': naa_na.element
    }
}

# Etch profiles
print("Generating etch test data...")
etch_linear = simulator.simulate_etch_profile(model="linear", nominal_rate=100, alpha=0.3)
etch_exp = simulator.simulate_etch_profile(model="exponential", nominal_rate=100, alpha=0.4)

test_data['etch'] = {
    'linear_loading': {
        'pattern_density': etch_linear.pattern_density.tolist(),
        'etch_rate': etch_linear.etch_rate.tolist()
    },
    'exponential_loading': {
        'pattern_density': etch_exp.pattern_density.tolist(),
        'etch_rate': etch_exp.etch_rate.tolist()
    }
}

# Save to file
with open('test_data/session12_test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print("Test data generated successfully!")
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Test data generated"
    else
        echo -e "${YELLOW}!${NC} Test data generation had issues"
    fi
}

################################################################################
# Validation
################################################################################

run_validation_tests() {
    echo -e "${YELLOW}→${NC} Running validation tests..."
    
    python3 -m pytest test_session12_integration.py -v --tb=short
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} All validation tests passed"
    else
        echo -e "${RED}✗${NC} Some validation tests failed"
        echo "  Review test output above"
    fi
}

################################################################################
# Main Deployment Flow
################################################################################

main() {
    echo ""
    echo "Starting deployment..."
    echo ""
    
    # Pre-flight checks
    check_dependencies
    check_python_packages
    
    # Database setup
    create_database_schema
    load_nuclear_data
    load_default_calibrations
    
    # Service configuration
    create_service_config
    create_start_script
    create_stop_script
    
    # Create directories
    mkdir -p test_data
    mkdir -p logs
    
    # Generate test data
    generate_test_data
    
    # Run validation
    if [ "$1" != "--skip-tests" ]; then
        run_validation_tests
    fi
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Session 12 Deployment Complete!                      ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Start services: ./start_session12_services.sh"
    echo "  2. Check API docs: http://localhost:8012/docs"
    echo "  3. Run tests: python3 -m pytest test_session12_integration.py"
    echo "  4. Stop services: ./stop_session12_services.sh"
    echo ""
    echo "Database tables created:"
    echo "  • sims_measurements, sims_profiles, sims_calibrations"
    echo "  • rbs_measurements, rbs_spectra, rbs_layers, rbs_fits"
    echo "  • naa_measurements, naa_decay_curves, naa_quantifications"
    echo "  • etch_measurements, etch_profiles, etch_loading_effects"
    echo ""
}

# Execute main deployment
main "$@"
