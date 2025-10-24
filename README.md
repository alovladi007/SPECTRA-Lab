# SPECTRA-Lab: Semiconductor Characterization Platform

Enterprise-grade semiconductor characterization platform with 40+ measurement methods.

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
├── apps/web/                  # Next.js frontend
├── services/
│   ├── instruments/           # Instrument control service
│   │   └── app/
│   │       ├── drivers/      # Instrument drivers
│   │       │   ├── core/     # VISA/SCPI core
│   │       │   └── builtin/  # Reference drivers
│   │       └── models/       # Database models
│   └── analysis/              # Analysis service
├── src/
│   ├── backend/models/        # Shared data models
│   └── drivers/               # Additional drivers
├── scripts/                   # Utility scripts
│   └── dev/                   # Development scripts
├── docs/                      # Documentation
├── infra/docker/              # Docker configuration
└── tests/                     # Test suites
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

## Project Status

**Sessions Completed:**
- ✅ Session 1: Setup & Architecture
- ✅ Session 2: Data Model & API
- ✅ Session 3: Instrument SDK & HIL
- ✅ Session 4: Electrical I (4PP & Hall Effect)

**Current:** Session 5 - Electrical II (I-V, C-V)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Proprietary - All rights reserved.
