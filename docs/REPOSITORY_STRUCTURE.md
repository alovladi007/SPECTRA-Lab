# SemiconductorLab - Repository Structure

semiconductorlab/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Main CI pipeline
│   │   ├── release.yml               # Release automation
│   │   ├── security-scan.yml         # Dependency scanning
│   │   └── deploy-staging.yml        # Auto-deploy to staging
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── method_implementation.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── apps/
│   ├── web/                          # Next.js frontend
│   │   ├── app/                      # App router
│   │   │   ├── (auth)/
│   │   │   │   ├── login/
│   │   │   │   │   └── page.tsx
│   │   │   │   └── register/
│   │   │   │       └── page.tsx
│   │   │   ├── dashboard/
│   │   │   │   └── page.tsx
│   │   │   ├── projects/
│   │   │   │   ├── page.tsx
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx
│   │   │   ├── samples/
│   │   │   │   ├── page.tsx
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx
│   │   │   ├── experiments/
│   │   │   │   ├── page.tsx          # Experiment list
│   │   │   │   ├── new/
│   │   │   │   │   └── page.tsx      # Experiment builder
│   │   │   │   └── [id]/
│   │   │   │       ├── page.tsx      # Experiment details
│   │   │   │       └── run/
│   │   │   │           └── page.tsx  # Live run viewer
│   │   │   ├── results/
│   │   │   │   ├── page.tsx
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx
│   │   │   ├── spc/
│   │   │   │   └── page.tsx          # SPC dashboard
│   │   │   ├── vm/
│   │   │   │   ├── page.tsx          # VM studio
│   │   │   │   ├── models/
│   │   │   │   │   └── page.tsx
│   │   │   │   └── train/
│   │   │   │       └── page.tsx
│   │   │   ├── admin/
│   │   │   │   ├── instruments/
│   │   │   │   │   └── page.tsx
│   │   │   │   ├── users/
│   │   │   │   │   └── page.tsx
│   │   │   │   └── calibrations/
│   │   │   │       └── page.tsx
│   │   │   ├── layout.tsx            # Root layout
│   │   │   └── page.tsx              # Home page
│   │   ├── components/
│   │   │   ├── ui/                   # shadcn/ui components
│   │   │   │   ├── button.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   ├── card.tsx
│   │   │   │   ├── dialog.tsx
│   │   │   │   └── ...
│   │   │   ├── charts/
│   │   │   │   ├── IVCurvePlot.tsx
│   │   │   │   ├── SpectrumPlot.tsx
│   │   │   │   ├── WaferMap.tsx
│   │   │   │   └── ControlChart.tsx
│   │   │   ├── forms/
│   │   │   │   ├── ExperimentForm.tsx
│   │   │   │   ├── SampleForm.tsx
│   │   │   │   └── MethodSelector.tsx
│   │   │   ├── layouts/
│   │   │   │   ├── MainLayout.tsx
│   │   │   │   ├── DashboardLayout.tsx
│   │   │   │   └── Sidebar.tsx
│   │   │   └── shared/
│   │   │       ├── DataTable.tsx
│   │   │       ├── FileUpload.tsx
│   │   │       └── StatusBadge.tsx
│   │   ├── lib/
│   │   │   ├── api.ts                # API client
│   │   │   ├── auth.ts               # Auth helpers
│   │   │   ├── utils.ts              # Utilities
│   │   │   └── constants.ts
│   │   ├── hooks/
│   │   │   ├── useAuth.ts
│   │   │   ├── useExperiment.ts
│   │   │   └── useSSE.ts             # Server-Sent Events
│   │   ├── stores/
│   │   │   ├── authStore.ts          # Zustand store
│   │   │   └── experimentStore.ts
│   │   ├── types/
│   │   │   ├── api.ts                # API types
│   │   │   ├── experiment.ts
│   │   │   └── user.ts
│   │   ├── public/
│   │   │   ├── images/
│   │   │   └── icons/
│   │   ├── .env.local
│   │   ├── .env.example
│   │   ├── next.config.js
│   │   ├── tailwind.config.ts
│   │   ├── tsconfig.json
│   │   ├── package.json
│   │   └── README.md
│   │
│   └── docs/                         # Documentation site (optional)
│       └── ...
│
├── services/
│   ├── auth/                         # Authentication service
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py               # FastAPI app
│   │   │   ├── config.py             # Settings
│   │   │   ├── database.py           # DB connection
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── user.py
│   │   │   │   └── session.py
│   │   │   ├── schemas/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── user.py
│   │   │   │   └── token.py
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth.py
│   │   │   │   └── users.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth_service.py
│   │   │   │   └── user_service.py
│   │   │   └── dependencies.py       # FastAPI dependencies
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── conftest.py
│   │   │   ├── test_auth.py
│   │   │   └── test_users.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── requirements-dev.txt
│   │   ├── pyproject.toml
│   │   └── README.md
│   │
│   ├── instruments/                  # Instrument service
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instrument.py
│   │   │   │   ├── run.py
│   │   │   │   ├── calibration.py
│   │   │   │   └── measurement.py
│   │   │   ├── schemas/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instrument.py
│   │   │   │   ├── run.py
│   │   │   │   └── measurement.py
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instruments.py
│   │   │   │   ├── runs.py
│   │   │   │   └── calibrations.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instrument_service.py
│   │   │   │   ├── acquisition_service.py
│   │   │   │   └── calibration_service.py
│   │   │   ├── drivers/              # Instrument drivers
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py           # Abstract base class
│   │   │   │   ├── connection.py     # Connection managers
│   │   │   │   ├── smu/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── keithley_2400.py
│   │   │   │   │   └── keithley_2600.py
│   │   │   │   ├── spectrometer/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── ocean_optics.py
│   │   │   │   │   └── avantes.py
│   │   │   │   ├── ellipsometer/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── ja_woollam.py
│   │   │   │   └── generic_scpi.py   # Fallback driver
│   │   │   ├── simulators/           # HIL simulators
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_simulator.py
│   │   │   │   ├── smu_simulator.py
│   │   │   │   ├── spectrometer_simulator.py
│   │   │   │   └── ellipsometer_simulator.py
│   │   │   ├── plugins/              # User-added drivers
│   │   │   │   └── README.md
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── visa_helper.py
│   │   │       └── data_validator.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── conftest.py
│   │   │   ├── test_drivers.py
│   │   │   └── test_simulators.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── requirements-dev.txt
│   │   └── README.md
│   │
│   ├── analysis/                     # Analysis service
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── analysis_result.py
│   │   │   │   └── model.py          # ML models
│   │   │   ├── schemas/
│   │   │   │   ├── __init__.py
│   │   │   │   └── analysis.py
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── analysis.py
│   │   │   │   ├── spc.py
│   │   │   │   └── ml.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   └── analysis_service.py
│   │   │   ├── methods/              # Analysis methods
│   │   │   │   ├── __init__.py
│   │   │   │   ├── electrical/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── four_point_probe.py
│   │   │   │   │   ├── hall_effect.py
│   │   │   │   │   ├── iv_analysis.py
│   │   │   │   │   ├── cv_analysis.py
│   │   │   │   │   └── dlts.py
│   │   │   │   ├── optical/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── uv_vis_nir.py
│   │   │   │   │   ├── ftir.py
│   │   │   │   │   ├── ellipsometry.py
│   │   │   │   │   ├── pl_el.py
│   │   │   │   │   └── raman.py
│   │   │   │   ├── structural/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── xrd.py
│   │   │   │   │   ├── sem_tem.py
│   │   │   │   │   └── afm.py
│   │   │   │   └── chemical/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── xps_xrf.py
│   │   │   │       ├── sims.py
│   │   │   │       └── rbs.py
│   │   │   ├── spc/                  # SPC algorithms
│   │   │   │   ├── __init__.py
│   │   │   │   ├── control_charts.py
│   │   │   │   ├── capability.py
│   │   │   │   └── rules.py
│   │   │   ├── ml/                   # ML models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── virtual_metrology.py
│   │   │   │   ├── anomaly_detection.py
│   │   │   │   └── drift_detection.py
│   │   │   ├── utils/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── curve_fitting.py
│   │   │   │   ├── statistics.py
│   │   │   │   └── units.py
│   │   │   └── workers/              # Celery tasks
│   │   │       ├── __init__.py
│   │   │       └── tasks.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── conftest.py
│   │   │   ├── test_methods/
│   │   │   │   ├── test_iv_analysis.py
│   │   │   │   └── ...
│   │   │   ├── test_spc.py
│   │   │   └── test_ml.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── requirements-dev.txt
│   │   └── README.md
│   │
│   ├── reporting/                    # Report service
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   └── reports.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   └── report_service.py
│   │   │   ├── templates/            # Jinja2 templates
│   │   │   │   ├── base.html
│   │   │   │   ├── run_report.html
│   │   │   │   └── batch_report.html
│   │   │   └── generators/
│   │   │       ├── __init__.py
│   │   │       ├── pdf_generator.py
│   │   │       ├── csv_exporter.py
│   │   │       └── matlab_exporter.py
│   │   ├── tests/
│   │   │   └── test_generators.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   └── notifications/                # Notification service
│       ├── app/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── config.py
│       │   ├── routers/
│       │   │   ├── __init__.py
│       │   │   └── notifications.py
│       │   ├── services/
│       │   │   ├── __init__.py
│       │   │   └── notification_service.py
│       │   └── channels/
│       │       ├── __init__.py
│       │       ├── email.py
│       │       ├── slack.py
│       │       └── webhook.py
│       ├── tests/
│       │   └── test_channels.py
│       ├── Dockerfile
│       ├── requirements.txt
│       └── README.md
│
├── packages/                         # Shared libraries
│   ├── common/                       # Shared Python utilities
│   │   ├── semiconductorlab_common/
│   │   │   ├── __init__.py
│   │   │   ├── types.py              # Common types
│   │   │   ├── units.py              # Pint-based units
│   │   │   ├── constants.py
│   │   │   └── utils.py
│   │   ├── tests/
│   │   │   └── test_units.py
│   │   ├── setup.py
│   │   └── README.md
│   │
│   └── types/                        # Shared TypeScript types
│       ├── src/
│       │   ├── index.ts
│       │   ├── api.ts
│       │   ├── experiment.ts
│       │   └── user.ts
│       ├── tsconfig.json
│       ├── package.json
│       └── README.md
│
├── data/                             # Test data and seeds
│   ├── seeds/
│   │   ├── users.json
│   │   ├── instruments.json
│   │   ├── samples.json
│   │   └── runs.json
│   ├── test_data/
│   │   ├── electrical/
│   │   │   ├── iv_diode.csv
│   │   │   ├── hall_si.csv
│   │   │   └── ...
│   │   ├── optical/
│   │   │   ├── uv_vis_gan.csv
│   │   │   └── ...
│   │   └── ...
│   └── golden_samples/               # Reference datasets
│       ├── si_resistivity.json
│       ├── gaas_mobility.json
│       └── ...
│
├── db/                               # Database migrations
│   ├── migrations/
│   │   ├── 001_initial_schema.sql
│   │   ├── 002_add_calibrations.sql
│   │   └── ...
│   ├── alembic/                      # Alembic migrations (Python)
│   │   ├── versions/
│   │   │   └── ...
│   │   ├── env.py
│   │   └── alembic.ini
│   └── README.md
│
├── infra/                            # Infrastructure as Code
│   ├── docker/
│   │   ├── docker-compose.yml        # Development environment
│   │   ├── docker-compose.prod.yml   # Production-like local
│   │   └── .env.example
│   ├── kubernetes/
│   │   └── helm/
│   │       └── semiconductorlab/
│   │           ├── Chart.yaml
│   │           ├── values.yaml
│   │           ├── values-dev.yaml
│   │           ├── values-staging.yaml
│   │           ├── values-prod.yaml
│   │           └── templates/
│   │               ├── web-deployment.yaml
│   │               ├── instruments-deployment.yaml
│   │               ├── analysis-deployment.yaml
│   │               ├── postgres-statefulset.yaml
│   │               ├── redis-statefulset.yaml
│   │               ├── nats-statefulset.yaml
│   │               ├── ingress.yaml
│   │               ├── hpa.yaml
│   │               ├── pvc.yaml
│   │               ├── secrets.yaml
│   │               └── ...
│   ├── terraform/                    # Optional: for cloud infra
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── monitoring/
│       ├── prometheus/
│       │   └── prometheus.yml
│       ├── grafana/
│       │   ├── dashboards/
│       │   │   ├── system_overview.json
│       │   │   ├── instrument_metrics.json
│       │   │   └── spc_dashboard.json
│       │   └── datasources/
│       │       └── prometheus.yml
│       └── loki/
│           └── loki-config.yml
│
├── docs/                             # Documentation
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── c4_diagrams.md
│   │   └── security.md
│   ├── guides/
│   │   ├── admin_guide.md
│   │   ├── user_guide.md
│   │   └── developer_guide.md
│   ├── methods/                      # Method playbooks
│   │   ├── electrical/
│   │   │   ├── four_point_probe.md
│   │   │   ├── hall_effect.md
│   │   │   ├── iv_characterization.md
│   │   │   └── ...
│   │   ├── optical/
│   │   │   └── ...
│   │   └── ...
│   ├── api/
│   │   ├── openapi.yaml              # OpenAPI specification
│   │   └── examples/
│   │       └── curl_examples.sh
│   └── tutorials/
│       ├── getting_started.md
│       ├── first_experiment.md
│       └── advanced_analysis.md
│
├── scripts/                          # Utility scripts
│   ├── dev/
│   │   ├── setup.sh                  # Initial setup
│   │   ├── seed_db.py                # Seed database
│   │   └── generate_test_data.py     # Generate synthetic data
│   ├── ops/
│   │   ├── backup_db.sh              # Backup script
│   │   ├── restore_db.sh             # Restore script
│   │   └── health_check.sh
│   └── ci/
│       ├── run_tests.sh
│       └── deploy.sh
│
├── .gitignore
├── .dockerignore
├── .editorconfig
├── .prettierrc                       # Frontend formatting
├── .eslintrc.json                    # Frontend linting
├── pyproject.toml                    # Python project config (Ruff, Black)
├── Makefile                          # Common tasks
├── turbo.json                        # Turborepo config (if using Turborepo)
├── package.json                      # Root package.json (monorepo)
├── pnpm-workspace.yaml               # pnpm workspaces (or npm/yarn)
├── README.md                         # Project overview
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # Software license
└── CHANGELOG.md                      # Version history

-----

## Key Files Content

### Root `README.md`

# SemiconductorLab Platform

> Enterprise-grade semiconductor characterization platform

[![CI](https://github.com/org/semiconductorlab/actions/workflows/ci.yml/badge.svg)](...)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Quick Start

### Prerequisites
- Docker 24+ & Docker Compose
- Node.js 20+ & pnpm 9+
- Python 3.11+
- Make

### Development Setup
# Clone repository
git clone https://github.com/org/semiconductorlab.git
cd semiconductorlab

# Start all services
make dev-up

# Access application
# Web: http://localhost:3000
# API Docs: http://localhost:8000/docs
# Grafana: http://localhost:3001

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Admin Guide](docs/guides/admin_guide.md)
- [User Guide](docs/guides/user_guide.md)
- [API Reference](docs/api/openapi.yaml)

## Contributing

See <CONTRIBUTING.md>

## License

MIT License - see <LICENSE>

---

### `Makefile`
.PHONY: help dev-up dev-down dev-logs test lint format build deploy

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development
dev-up: ## Start development environment
	docker-compose -f infra/docker/docker-compose.yml up -d
	@echo "Services started. Web: http://localhost:3000"

dev-down: ## Stop development environment
	docker-compose -f infra/docker/docker-compose.yml down

dev-logs: ## Tail logs
	docker-compose -f infra/docker/docker-compose.yml logs -f

dev-reset: ## Reset database and start fresh
	docker-compose -f infra/docker/docker-compose.yml down -v
	make dev-up
	sleep 5
	make seed-db

# Database
migrate: ## Run database migrations
	cd services/instruments && alembic upgrade head

seed-db: ## Seed database with test data
	python scripts/dev/seed_db.py

# Testing
test: ## Run all tests
	./scripts/ci/run_tests.sh

test-unit: ## Run unit tests
	cd services/instruments && pytest tests/ -v
	cd services/analysis && pytest tests/ -v

test-e2e: ## Run end-to-end tests
	cd apps/web && pnpm test:e2e

# Linting & Formatting
lint: ## Lint all code
	cd apps/web && pnpm lint
	cd services/instruments && ruff check .
	cd services/analysis && ruff check .

format: ## Format all code
	cd apps/web && pnpm format
	cd services/instruments && ruff format .
	cd services/analysis && ruff format .

# Build
build: ## Build all Docker images
	docker-compose -f infra/docker/docker-compose.yml build

# Deployment (example - customize per environment)
deploy-staging: ## Deploy to staging
	helm upgrade --install semiconductorlab infra/kubernetes/helm/semiconductorlab \
		--namespace staging --create-namespace \
		--values infra/kubernetes/helm/semiconductorlab/values-staging.yaml

deploy-prod: ## Deploy to production (requires confirmation)
	@read -p "Deploy to PRODUCTION? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	helm upgrade --install semiconductorlab infra/kubernetes/helm/semiconductorlab \
		--namespace production --create-namespace \
		--values infra/kubernetes/helm/semiconductorlab/values-prod.yaml

# Cleanup
clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	cd apps/web && rm -rf .next node_modules

-----

### `.gitignore`

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/
.pytest_cache/

# Node
node_modules/
.next/
out/
.turbo/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Environment
.env
.env.local
.env.*.local
*.env

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Database
*.db
*.sqlite

# Logs
logs/
*.log

# Docker
.docker/

# Secrets
secrets/
*.pem
*.key

-----

### `pyproject.toml` (Root)

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "B", "C90"]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.black]
line-length = 120
target-version = ['py311']

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
pythonpath = ["services"]

-----

### `package.json` (Root)

{
  "name": "semiconductorlab",
  "version": "1.0.0",
  "private": true,
  "workspaces": [
    "apps/*",
    "packages/*"
  ],
  "scripts": {
    "dev": "turbo run dev",
    "build": "turbo run build",
    "test": "turbo run test",
    "lint": "turbo run lint",
    "format": "prettier --write \"**/*.{ts,tsx,md,json}\""
  },
  "devDependencies": {
    "prettier": "^3.1.0",
    "turbo": "^1.11.0"
  },
  "engines": {
    "node": ">=20",
    "pnpm": ">=9"
  }
}

-----

### `turbo.json`

{
  "$schema": "https://turbo.build/schema.json",
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "dist/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {
      "dependsOn": ["^lint"]
    },
    "test": {
      "dependsOn": ["^build"],
      "outputs": ["coverage/**"]
    }
  }
}

-----

## Setup Instructions

### 1. Clone and Initialize

git clone https://github.com/org/semiconductorlab.git
cd semiconductorlab

# Install frontend dependencies
pnpm install

# Create Python virtual environments (per service)
cd services/instruments
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

# Repeat for other services...

### 2. Configure Environment

# Copy example env files
cp apps/web/.env.example apps/web/.env.local
cp infra/docker/.env.example infra/docker/.env

# Edit .env files with your settings

### 3. Start Development Environment

make dev-up

### 4. Initialize Database

# Run migrations
make migrate

# Seed with test data
make seed-db

### 5. Verify Installation

# Check all services are running
docker ps

# Access services:
# Web UI: http://localhost:3000
# API Gateway: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Grafana: http://localhost:3001 (admin/admin)

-----

*End of Repository Scaffold*