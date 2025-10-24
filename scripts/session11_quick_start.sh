#!/bin/bash

#######################################################################
# Session 11: XPS/XRF Analysis - Quick Start Script
# One-command setup and launch
#######################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Session 11: XPS/XRF Analysis Quick Start    ${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check Python
echo -e "${GREEN}[1/6]${NC} Checking Python installation..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | grep -Po '\d+\.\d+')
    echo "✓ Python $python_version found"
else
    echo "✗ Python not found. Please install Python 3.8+"
    exit 1
fi

# Check Node.js
echo -e "${GREEN}[2/6]${NC} Checking Node.js installation..."
if command -v node &> /dev/null; then
    node_version=$(node --version)
    echo "✓ Node.js $node_version found"
else
    echo "⚠ Node.js not found (optional for UI)"
fi

# Install Python dependencies
echo -e "${GREEN}[3/6]${NC} Installing Python dependencies..."
pip install -q -r requirements.txt 2>/dev/null || {
    echo "⚠ Some Python packages failed to install"
    echo "  Run: pip install -r requirements.txt"
}
echo "✓ Python packages installed"

# Install Node dependencies (if Node exists)
if command -v npm &> /dev/null; then
    echo -e "${GREEN}[4/6]${NC} Installing Node.js dependencies..."
    npm install --silent 2>/dev/null || {
        echo "⚠ Some Node packages failed to install"
        echo "  Run: npm install"
    }
    echo "✓ Node packages installed"
else
    echo -e "${GREEN}[4/6]${NC} Skipping Node.js dependencies (Node not found)"
fi

# Generate sample data
echo -e "${GREEN}[5/6]${NC} Generating sample data..."
python3 << 'EOF'
import sys, os
sys.path.insert(0, 'src')
import numpy as np

# Simple XPS spectrum
be = np.linspace(0, 1200, 2400)
intensity = np.random.normal(1000, 100, len(be))
intensity += 5000 * np.exp(-((be - 284.5)**2) / 2)  # C 1s
intensity += 3000 * np.exp(-((be - 532.5)**2) / 3)  # O 1s
np.savetxt('data/sample_xps.txt', np.column_stack((be, intensity)), 
           fmt='%.2f', header='BE(eV) Intensity')

# Simple XRF spectrum  
energy = np.linspace(0.1, 20, 2000)
counts = np.random.poisson(100, len(energy))
counts = counts + 5000 * np.exp(-((energy - 1.74)**2) / 0.001)  # Si
counts = counts + 3000 * np.exp(-((energy - 6.404)**2) / 0.002)  # Fe
np.savetxt('data/sample_xrf.txt', np.column_stack((energy, counts)),
           fmt='%.3f %.0f', header='Energy(keV) Counts')

print("✓ Sample data generated")
EOF

# Create run script
echo -e "${GREEN}[6/6]${NC} Creating run script..."
cat > run_analysis.py << 'EOF'
#!/usr/bin/env python3
"""Quick test of XPS/XRF analysis system"""

import sys
import os
sys.path.insert(0, 'src')

from chemical_analyzer import XPSAnalyzer, XRFAnalyzer, ChemicalSimulator
import numpy as np

print("\n=== Testing XPS/XRF Analysis System ===\n")

# Test XPS
print("1. Testing XPS Analysis...")
xps = XPSAnalyzer()
data = np.loadtxt('data/sample_xps.txt')
be, intensity = data[:, 0], data[:, 1]
be_proc, int_proc = xps.process_spectrum(be, intensity)
peaks = xps.find_peaks(be_proc, int_proc)
print(f"   Found {len(peaks)} peaks")
for peak in peaks[:3]:
    print(f"   - {peak.get('element', 'Unknown')} at {peak['position']:.1f} eV")

# Test XRF
print("\n2. Testing XRF Analysis...")
xrf = XRFAnalyzer()
data = np.loadtxt('data/sample_xrf.txt')
energy, counts = data[:, 0], data[:, 1]
energy_proc, counts_proc = xrf.process_spectrum(energy, counts)
peaks = xrf.find_peaks(energy_proc, counts_proc)
print(f"   Found {len(peaks)} peaks")
for peak in peaks[:3]:
    print(f"   - {peak.get('element_line', 'Unknown')} at {peak['energy']:.3f} keV")

print("\n✓ Analysis system is working!\n")
print("To start the full system:")
print("  - Backend API: uvicorn src.api:app --reload --port 8011")
print("  - Frontend UI: npm run dev (in ui/ directory)")
EOF

chmod +x run_analysis.py

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}✓ Quick Start Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Directory structure:"
echo "  src/           - Core Python implementation"
echo "  ui/            - React user interface"
echo "  tests/         - Integration tests"
echo "  config/        - Configuration files"
echo "  data/          - Sample data files"
echo "  deployment/    - Deployment scripts"
echo "  docs/          - Documentation"
echo ""
echo "To test the system:"
echo "  python3 run_analysis.py"
echo ""
echo "To run full tests:"
echo "  python3 -m pytest tests/ -v"
echo ""
echo "To start services:"
echo "  Backend: uvicorn src.api:app --reload --port 8011"
echo "  Frontend: cd ui && npm run dev"
echo ""
