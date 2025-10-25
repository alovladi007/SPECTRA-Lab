#!/bin/bash

###############################################################################
# Session 11: Surface Analysis (XPS/XRF) - Deployment Script
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SERVICE_NAME="surface-analysis"
SERVICE_PORT=8011
DB_NAME="semiconductor_lab"
DB_USER="lab_user"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Session 11: Surface Analysis Deployment             ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

create_database_schema() {
    echo -e "${YELLOW}→${NC} Creating database schema..."
    
    cat > /tmp/session11_schema.sql << 'EOF'
-- Session 11: Surface Analysis - Database Schema

-- XPS measurements
CREATE TABLE IF NOT EXISTS xps_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID NOT NULL,
    run_id UUID NOT NULL,
    scan_type VARCHAR(20) NOT NULL,
    x_ray_source VARCHAR(20) DEFAULT 'Al Kα',
    x_ray_energy FLOAT DEFAULT 1486.6,
    pass_energy FLOAT DEFAULT 20.0,
    dwell_time FLOAT DEFAULT 0.1,
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operator VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- XPS spectra
CREATE TABLE IF NOT EXISTS xps_spectra (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES xps_measurements(id) ON DELETE CASCADE,
    binding_energy FLOAT[],
    intensity FLOAT[],
    element VARCHAR(10),
    orbital VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- XPS peaks
CREATE TABLE IF NOT EXISTS xps_peaks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spectrum_id UUID REFERENCES xps_spectra(id) ON DELETE CASCADE,
    position FLOAT NOT NULL,
    area FLOAT,
    fwhm FLOAT,
    height FLOAT,
    element VARCHAR(10),
    orbital VARCHAR(10),
    oxidation_state VARCHAR(10),
    assignment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- XPS quantification
CREATE TABLE IF NOT EXISTS xps_quantifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spectrum_id UUID REFERENCES xps_spectra(id) ON DELETE CASCADE,
    composition JSONB NOT NULL,
    uncertainty JSONB,
    background_type VARCHAR(20),
    sensitivity_factors JSONB,
    total_area FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- XRF measurements
CREATE TABLE IF NOT EXISTS xrf_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sample_id UUID NOT NULL,
    run_id UUID NOT NULL,
    mode VARCHAR(20) DEFAULT 'EDXRF',
    tube_voltage FLOAT DEFAULT 40.0,
    tube_current FLOAT DEFAULT 0.8,
    live_time FLOAT DEFAULT 100.0,
    filter_type VARCHAR(20),
    atmosphere VARCHAR(20),
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operator VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- XRF spectra
CREATE TABLE IF NOT EXISTS xrf_spectra (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID REFERENCES xrf_measurements(id) ON DELETE CASCADE,
    energy FLOAT[],
    intensity FLOAT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- XRF peaks
CREATE TABLE IF NOT EXISTS xrf_peaks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spectrum_id UUID REFERENCES xrf_spectra(id) ON DELETE CASCADE,
    energy FLOAT NOT NULL,
    intensity FLOAT,
    area FLOAT,
    fwhm FLOAT,
    element VARCHAR(10),
    line VARCHAR(10),
    assignment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- XRF quantification
CREATE TABLE IF NOT EXISTS xrf_quantifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spectrum_id UUID REFERENCES xrf_spectra(id) ON DELETE CASCADE,
    composition JSONB NOT NULL,
    uncertainty JSONB,
    method VARCHAR(50),
    thickness FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_xps_measurements_sample ON xps_measurements(sample_id);
CREATE INDEX IF NOT EXISTS idx_xps_spectra_measurement ON xps_spectra(measurement_id);
CREATE INDEX IF NOT EXISTS idx_xrf_measurements_sample ON xrf_measurements(sample_id);
CREATE INDEX IF NOT EXISTS idx_xrf_spectra_measurement ON xrf_spectra(measurement_id);

-- Schema version
INSERT INTO schema_versions (module, version) 
VALUES ('session11_surface_analysis', '1.0.0')
ON CONFLICT (module) DO UPDATE SET 
    version = EXCLUDED.version,
    deployed_at = CURRENT_TIMESTAMP;
EOF

    psql -h localhost -U $DB_USER -d $DB_NAME -f /tmp/session11_schema.sql
    rm /tmp/session11_schema.sql
    echo -e "${GREEN}✓${NC} Database schema created"
}

create_start_script() {
    cat > start_session11_services.sh << 'EOF'
#!/bin/bash
echo "Starting Session 11 services..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export SESSION11_PORT=8011
python3 -m uvicorn session11_surface_analysis_complete_implementation:app \
    --host 0.0.0.0 \
    --port $SESSION11_PORT \
    --reload &
echo "Service started! API: http://localhost:$SESSION11_PORT/docs"
EOF
    chmod +x start_session11_services.sh
    echo -e "${GREEN}✓${NC} Start script created"
}

create_stop_script() {
    cat > stop_session11_services.sh << 'EOF'
#!/bin/bash
pkill -f "session11_surface_analysis"
echo "Session 11 services stopped"
EOF
    chmod +x stop_session11_services.sh
}

echo ""
create_database_schema
create_start_script
create_stop_script

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Session 11 Deployment Complete!                      ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Start services: ./start_session11_services.sh"
echo "API docs: http://localhost:8011/docs"
echo ""
