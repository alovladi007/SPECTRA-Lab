# SPECTRA-Lab: Semiconductor Characterization Platform

Enterprise-grade semiconductor characterization platform with comprehensive electrical, optical, and structural characterization capabilities.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- PostgreSQL 15+ (or use Docker)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/alovladi007/SPECTRA-Lab.git
cd SPECTRA-Lab
```

2. **Install Python dependencies**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Install Node.js dependencies** (for web UI)
```bash
cd apps/web
npm install  # or: pnpm install
cd ../..
```

4. **Start development environment** (with Docker)
```bash
make dev-up
```

Or without Docker:
```bash
# Start PostgreSQL (if not using Docker)
# Configure database connection in .env

# Run migrations
python -m alembic upgrade head

# Start API server
cd services/instruments
uvicorn app.main:app --reload --port 8000

# Start web UI (in another terminal)
cd apps/web
npm run dev
```

### Access

- **Web UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3001 (admin/admin)

## Repository Structure

```
SPECTRA-Lab/
├── apps/web/                     # Next.js frontend
│   └── src/app/(dashboard)/
│       ├── electrical/           # Electrical measurement UIs
│       ├── optical/              # Optical characterization UIs
│       ├── structural/           # Structural analysis UIs
│       ├── chemical/             # Chemical analysis UIs
│       ├── spc/                  # Statistical Process Control
│       └── ml/                   # Machine Learning & Virtual Metrology
├── services/
│   ├── instruments/              # Instrument control service
│   │   └── app/
│   │       ├── drivers/          # Instrument drivers
│   │       │   ├── core/         # VISA/SCPI core
│   │       │   └── builtin/      # Reference drivers
│   │       └── models/           # Database models (SQLAlchemy)
│   └── analysis/                 # Analysis service
│       └── app/methods/
│           ├── electrical/       # Electrical analysis modules
│           ├── optical/          # Optical analysis modules
│           ├── structural/       # Structural analysis modules
│           ├── chemical/         # Chemical analysis modules
│           ├── spc/              # Statistical Process Control
│           └── ml/               # Machine Learning & Virtual Metrology
├── src/
│   ├── backend/
│   │   ├── models/               # Pydantic schemas
│   │   └── services/             # Backend services
│   └── drivers/                  # Additional drivers & simulators
├── scripts/                      # Deployment & utility scripts
│   └── dev/                      # Development scripts
├── docs/                         # Complete documentation
│   ├── sessions/                 # Session implementation guides
│   ├── methods/                  # Method playbooks
│   └── api/                      # API specifications
├── infra/docker/                 # Docker configuration
├── tests/                        # Test suites
│   ├── integration/              # Integration tests
│   ├── unit/                     # Unit tests
│   └── validation/               # Validation scenarios
└── db/migrations/                # Database migrations
```

## Available Commands

```bash
make dev-up          # Start all services
make dev-down        # Stop all services
make dev-logs        # View logs
make dev-reset       # Reset environment (deletes data!)
make test            # Run tests
make lint            # Run linters
make format          # Format code
```

## Documentation

- [Master Roadmap](docs/ROADMAP.md)
- [Data Model](docs/DATA_MODEL_SPECIFICATION.md)
- [API Specification](docs/api/openapi_specification.yaml)
- [Session Guides](docs/sessions/)

## Characterization Capabilities

### Electrical Characterization
- ✅ **Four-Point Probe (4PP)** - Sheet resistance measurement
- ✅ **Hall Effect** - Carrier concentration and mobility
- ✅ **I-V Characterization** - Diode, transistor, and device curves
- ✅ **C-V Profiling** - Capacitance-voltage analysis
- ✅ **BJT Analysis** - Bipolar junction transistor characterization
- ✅ **MOSFET Analysis** - Metal-oxide-semiconductor FET testing
- ✅ **Solar Cell Testing** - Photovoltaic device characterization
- ✅ **DLTS** - Deep Level Transient Spectroscopy
- ✅ **EBIC** - Electron Beam Induced Current imaging
- ✅ **PCD** - Photoconductance Decay lifetime measurement

### Optical Characterization
- ✅ **UV-Vis-NIR Spectroscopy** - Absorption/transmission/reflectance
- ✅ **FTIR** - Fourier Transform Infrared Spectroscopy
- ✅ **Ellipsometry** - Thin film optical properties
- ✅ **Photoluminescence (PL)** - Optical emission analysis
- ✅ **Raman Spectroscopy** - Molecular vibrational analysis

### Structural Characterization
- ✅ **X-Ray Diffraction (XRD)** - Crystal structure and phase analysis
- ✅ **SEM (Scanning Electron Microscopy)** - High-resolution surface imaging
- ✅ **TEM (Transmission Electron Microscopy)** - Atomic-scale imaging
- ✅ **AFM (Atomic Force Microscopy)** - Surface topography and roughness
- ✅ **Optical Microscopy** - Multi-scale imaging and inspection

### Chemical Characterization

#### Surface Analysis
- ✅ **XPS (X-ray Photoelectron Spectroscopy)** - Surface chemistry and chemical states
- ✅ **XRF (X-ray Fluorescence)** - Elemental composition analysis

#### Bulk Analysis
- ✅ **SIMS (Secondary Ion Mass Spectrometry)** - Depth profiling and dopant quantification
- ✅ **RBS (Rutherford Backscattering)** - Multi-layer composition and thickness
- ✅ **NAA (Neutron Activation Analysis)** - Trace element detection
- ✅ **Chemical Etch Analysis** - Loading effect characterization

### Statistical Process Control (SPC)
- ✅ **Control Charts** - X-bar/R, I-MR, EWMA, CUSUM for real-time monitoring
- ✅ **Western Electric Rules** - All 8 rules for out-of-control detection
- ✅ **Process Capability** - Cp, Cpk, Pp, Ppk, Sigma Level, DPMO
- ✅ **Trend Analysis** - Linear regression with forecasting
- ✅ **Root Cause Analysis** - AI-assisted suggestions for violations
- ✅ **Real-time Alerts** - Severity-based with escalation

### Machine Learning & Virtual Metrology
- ✅ **Feature Engineering** - Automated feature generation (rolling stats, ratios, temporal)
- ✅ **Virtual Metrology** - Predict process metrics from equipment data (RF, GB, LightGBM)
- ✅ **Anomaly Detection** - Real-time anomaly identification (Isolation Forest, Elliptic Envelope)
- ✅ **Drift Detection** - Track model and data distribution drift (KS test, PSI, KL divergence)
- ✅ **Time Series Forecasting** - Predict future trends (Prophet, ARIMA)
- ✅ **AutoML** - Automated hyperparameter optimization (Optuna)
- ✅ **Model Explainability** - SHAP, LIME, permutation importance
- ✅ **Ensemble Methods** - Stacking, voting, blending for improved accuracy
- ✅ **A/B Testing** - Statistical significance testing for model comparison
- ✅ **Online Learning** - Incremental updates for continuous improvement
- ✅ **Model Registry** - Version control and lifecycle management
- ✅ **Production Monitoring** - Prometheus metrics, real-time alerting

## Project Status

**All Sessions Complete:**
- ✅ Session 1-2: Infrastructure & Architecture
- ✅ Session 3: Instrument SDK & HIL Simulators
- ✅ Session 4: Electrical I (4PP & Hall Effect)
- ✅ Session 5: Electrical II (I-V & C-V Characterization)
- ✅ Session 6: Electrical III (DLTS, EBIC, PCD)
- ✅ Session 7: Optical I (UV-Vis-NIR, FTIR)
- ✅ Session 8: Optical Advanced (Ellipsometry, PL, Raman)
- ✅ Session 9: Structural I (X-Ray Diffraction)
- ✅ Session 10: Structural II (Microscopy & Imaging)
- ✅ Session 11: Chemical I - Surface Analysis (XPS & XRF)
- ✅ Session 12: Chemical II - Bulk Analysis (SIMS, RBS, NAA, Etch)
- ✅ Session 13: Statistical Process Control (SPC Hub)
- ✅ Session 14: Machine Learning & Virtual Metrology (Enhanced)

**Total Files:** 175 integrated files
**Total Capabilities:** 26 characterization methods + SPC + ML/VM suite
**Status:** Enterprise Production Ready ✅

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Proprietary - All rights reserved.
