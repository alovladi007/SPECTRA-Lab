.PHONY: help dev-up dev-down dev-logs dev-reset migrate seed-db generate-test-data test lint format build clean

# Default target

.DEFAULT_GOAL := help

# Colors for output

BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
@echo “$(BLUE)SemiconductorLab Platform - Development Commands$(NC)”
@echo “”
@grep -E ‘^[a-zA-Z_-]+:.*?## .*$$’ $(MAKEFILE_LIST) | sort | awk ‘BEGIN {FS = “:.*?## “}; {printf “  $(GREEN)%-20s$(NC) %s\n”, $$1, $$2}’
@echo “”

# ============================================================================

# Development Environment

# ============================================================================

dev-up: ## Start all development services
@echo “$(BLUE)Starting SemiconductorLab development environment…$(NC)”
docker-compose -f infra/docker/docker-compose.yml up -d
@echo “$(GREEN)✓ Services started$(NC)”
@echo “”
@echo “Access points:”
@echo “  - Web UI:        http://localhost:3000”
@echo “  - API Docs:      http://localhost:8000/docs”
@echo “  - Analysis API:  http://localhost:8001/docs”
@echo “  - Grafana:       http://localhost:3001 (admin/admin)”
@echo “  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)”
@echo “  - Prometheus:    http://localhost:9090”
@echo “”

dev-down: ## Stop all development services
@echo “$(YELLOW)Stopping SemiconductorLab development environment…$(NC)”
docker-compose -f infra/docker/docker-compose.yml down
@echo “$(GREEN)✓ Services stopped$(NC)”

dev-logs: ## Tail logs from all services
docker-compose -f infra/docker/docker-compose.yml logs -f

dev-logs-service: ## Tail logs from specific service (usage: make dev-logs-service SERVICE=instruments)
@if [ -z “$(SERVICE)” ]; then   
echo “$(RED)Error: SERVICE parameter required$(NC)”;   
echo “Usage: make dev-logs-service SERVICE=instruments”;   
exit 1;   
fi
docker-compose -f infra/docker/docker-compose.yml logs -f $(SERVICE)

dev-reset: ## Reset entire development environment (WARNING: deletes all data)
@echo “$(RED)WARNING: This will delete all data. Press Ctrl+C to cancel, Enter to continue.$(NC)”
@read confirm
@echo “$(YELLOW)Resetting development environment…$(NC)”
docker-compose -f infra/docker/docker-compose.yml down -v
@echo “$(GREEN)✓ Volumes deleted$(NC)”
$(MAKE) dev-up
@echo “$(BLUE)Waiting for services to be ready…$(NC)”
sleep 10
$(MAKE) migrate
$(MAKE) seed-db
@echo “$(GREEN)✓ Development environment reset complete$(NC)”

dev-shell: ## Open shell in a service container (usage: make dev-shell SERVICE=instruments)
@if [ -z “$(SERVICE)” ]; then   
echo “$(RED)Error: SERVICE parameter required$(NC)”;   
echo “Usage: make dev-shell SERVICE=instruments”;   
exit 1;   
fi
docker-compose -f infra/docker/docker-compose.yml exec $(SERVICE) /bin/bash

# ============================================================================

# Database Management

# ============================================================================

migrate: ## Run database migrations
@echo “$(BLUE)Running database migrations…$(NC)”
docker-compose -f infra/docker/docker-compose.yml exec instruments alembic upgrade head
@echo “$(GREEN)✓ Migrations applied$(NC)”

migrate-rollback: ## Rollback last migration
@echo “$(YELLOW)Rolling back last migration…$(NC)”
docker-compose -f infra/docker/docker-compose.yml exec instruments alembic downgrade -1
@echo “$(GREEN)✓ Migration rolled back$(NC)”

migrate-create: ## Create new migration (usage: make migrate-create NAME=“add_new_table”)
@if [ -z “$(NAME)” ]; then   
echo “$(RED)Error: NAME parameter required$(NC)”;   
echo “Usage: make migrate-create NAME="add_new_table"”;   
exit 1;   
fi
docker-compose -f infra/docker/docker-compose.yml exec instruments alembic revision -m “$(NAME)”
@echo “$(GREEN)✓ Migration created$(NC)”

seed-db: ## Seed database with initial data
@echo “$(BLUE)Seeding database…$(NC)”
docker-compose -f infra/docker/docker-compose.yml exec instruments python /app/scripts/dev/seed_db.py
@echo “$(GREEN)✓ Database seeded$(NC)”

db-shell: ## Open PostgreSQL shell
docker-compose -f infra/docker/docker-compose.yml exec postgres psql -U postgres -d semiconductorlab_dev

db-backup: ## Backup database
@echo “$(BLUE)Backing up database…$(NC)”
mkdir -p backups
docker-compose -f infra/docker/docker-compose.yml exec -T postgres pg_dump -U postgres semiconductorlab_dev > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
@echo “$(GREEN)✓ Database backed up to backups/$(NC)”

db-restore: ## Restore database from backup (usage: make db-restore FILE=backups/backup_20251021_120000.sql)
@if [ -z “$(FILE)” ]; then   
echo “$(RED)Error: FILE parameter required$(NC)”;   
echo “Usage: make db-restore FILE=backups/backup_20251021_120000.sql”;   
exit 1;   
fi
@echo “$(YELLOW)Restoring database from $(FILE)…$(NC)”
docker-compose -f infra/docker/docker-compose.yml exec -T postgres psql -U postgres semiconductorlab_dev < $(FILE)
@echo “$(GREEN)✓ Database restored$(NC)”

# ============================================================================

# Test Data Generation

# ============================================================================

generate-test-data: ## Generate synthetic test data for all methods
@echo “$(BLUE)Generating test data…$(NC)”
python scripts/dev/generate_test_data.py
@echo “$(GREEN)✓ Test data generated in data/test_data/$(NC)”

generate-test-data-method: ## Generate test data for specific method (usage: make generate-test-data-method METHOD=iv_sweep)
@if [ -z “$(METHOD)” ]; then   
echo “$(RED)Error: METHOD parameter required$(NC)”;   
echo “Usage: make generate-test-data-method METHOD=iv_sweep”;   
exit 1;   
fi
python scripts/dev/generate_test_data.py –method $(METHOD)

# ============================================================================

# Testing

# ============================================================================

test: ## Run all tests
@echo “$(BLUE)Running all tests…$(NC)”
$(MAKE) test-unit
$(MAKE) test-integration
@echo “$(GREEN)✓ All tests passed$(NC)”

test-unit: ## Run unit tests
@echo “$(BLUE)Running unit tests…$(NC)”
@echo “$(YELLOW)Backend services:$(NC)”
cd services/instruments && pytest tests/ -v –cov=app –cov-report=term-missing
cd services/analysis && pytest tests/ -v –cov=app –cov-report=term-missing
@echo “$(YELLOW)Common package:$(NC)”
cd packages/common && pytest tests/ -v –cov=semiconductorlab_common –cov-report=term-missing
@echo “$(GREEN)✓ Unit tests passed$(NC)”

test-integration: ## Run integration tests
@echo “$(BLUE)Running integration tests…$(NC)”
cd services/instruments && pytest tests/integration/ -v
@echo “$(GREEN)✓ Integration tests passed$(NC)”

test-e2e: ## Run end-to-end tests
@echo “$(BLUE)Running end-to-end tests…$(NC)”
cd apps/web && pnpm test:e2e
@echo “$(GREEN)✓ E2E tests passed$(NC)”

test-coverage: ## Generate HTML coverage report
@echo “$(BLUE)Generating coverage report…$(NC)”
cd services/instruments && pytest tests/ –cov=app –cov-report=html
cd services/analysis && pytest tests/ –cov=app –cov-report=html
@echo “$(GREEN)✓ Coverage reports generated$(NC)”
@echo “Open services/instruments/htmlcov/index.html in browser”

# ============================================================================

# Code Quality

# ============================================================================

lint: ## Lint all code
@echo “$(BLUE)Linting code…$(NC)”
@echo “$(YELLOW)Backend (Python):$(NC)”
cd services/instruments && ruff check .
cd services/analysis && ruff check .
cd packages/common && ruff check .
@echo “$(YELLOW)Frontend (TypeScript):$(NC)”
cd apps/web && pnpm lint
@echo “$(GREEN)✓ Linting complete$(NC)”

format: ## Format all code
@echo “$(BLUE)Formatting code…$(NC)”
@echo “$(YELLOW)Backend (Python):$(NC)”
cd services/instruments && ruff format .
cd services/analysis && ruff format .
cd packages/common && ruff format .
@echo “$(YELLOW)Frontend (TypeScript):$(NC)”
cd apps/web && pnpm format
@echo “$(GREEN)✓ Formatting complete$(NC)”

typecheck: ## Run type checking
@echo “$(BLUE)Running type checks…$(NC)”
@echo “$(YELLOW)Backend (mypy):$(NC)”
cd services/instruments && mypy app/
cd services/analysis && mypy app/
@echo “$(YELLOW)Frontend (tsc):$(NC)”
cd apps/web && pnpm type-check
@echo “$(GREEN)✓ Type checking complete$(NC)”

# ============================================================================

# Build & Deploy

# ============================================================================

build: ## Build all Docker images
@echo “$(BLUE)Building Docker images…$(NC)”
docker-compose -f infra/docker/docker-compose.yml build
@echo “$(GREEN)✓ Images built$(NC)”

build-service: ## Build specific service (usage: make build-service SERVICE=instruments)
@if [ -z “$(SERVICE)” ]; then   
echo “$(RED)Error: SERVICE parameter required$(NC)”;   
echo “Usage: make build-service SERVICE=instruments”;   
exit 1;   
fi
docker-compose -f infra/docker/docker-compose.yml build $(SERVICE)

deploy-staging: ## Deploy to staging environment
@echo “$(BLUE)Deploying to staging…$(NC)”
helm upgrade –install semiconductorlab infra/kubernetes/helm/semiconductorlab   
–namespace staging –create-namespace   
–values infra/kubernetes/helm/semiconductorlab/values-staging.yaml
@echo “$(GREEN)✓ Deployed to staging$(NC)”

deploy-prod: ## Deploy to production (requires confirmation)
@echo “$(RED)WARNING: Deploying to PRODUCTION$(NC)”
@echo “Press Ctrl+C to cancel, Enter to continue.”
@read confirm
@echo “$(BLUE)Deploying to production…$(NC)”
helm upgrade –install semiconductorlab infra/kubernetes/helm/semiconductorlab   
–namespace production –create-namespace   
–values infra/kubernetes/helm/semiconductorlab/values-prod.yaml
@echo “$(GREEN)✓ Deployed to production$(NC)”

# ============================================================================

# Maintenance & Cleanup

# ============================================================================

clean: ## Clean build artifacts and cache
@echo “$(YELLOW)Cleaning build artifacts…$(NC)”
find . -type d -name “**pycache**” -exec rm -rf {} + 2>/dev/null || true
find . -type d -name “*.egg-info” -exec rm -rf {} + 2>/dev/null || true
find . -type d -name “.pytest_cache” -exec rm -rf {} + 2>/dev/null || true
find . -type d -name “.mypy_cache” -exec rm -rf {} + 2>/dev/null || true
find . -type d -name “.ruff_cache” -exec rm -rf {} + 2>/dev/null || true
find . -type d -name “htmlcov” -exec rm -rf {} + 2>/dev/null || true
find . -type f -name “*.pyc” -delete 2>/dev/null || true
find . -type f -name “.coverage” -delete 2>/dev/null || true
@echo “$(YELLOW)Cleaning Node.js artifacts…$(NC)”
find apps/web -type d -name “node_modules” -exec rm -rf {} + 2>/dev/null || true
find apps/web -type d -name “.next” -exec rm -rf {} + 2>/dev/null || true
@echo “$(GREEN)✓ Cleanup complete$(NC)”

clean-docker: ## Clean Docker resources (images, containers, volumes)
@echo “$(RED)WARNING: This will remove all Docker resources. Press Ctrl+C to cancel, Enter to continue.$(NC)”
@read confirm
@echo “$(YELLOW)Cleaning Docker resources…$(NC)”
docker-compose -f infra/docker/docker-compose.yml down -v –rmi all
docker system prune -af –volumes
@echo “$(GREEN)✓ Docker cleanup complete$(NC)”

# ============================================================================

# Documentation

# ============================================================================

docs-serve: ## Serve documentation locally
@echo “$(BLUE)Starting documentation server…$(NC)”
cd docs && python -m http.server 8080
@echo “$(GREEN)Documentation available at http://localhost:8080$(NC)”

docs-build: ## Build API documentation
@echo “$(BLUE)Building API documentation…$(NC)”
cd services/instruments && pydoc-markdown -p app –render-toc > ../../docs/api/instruments.md
cd services/analysis && pydoc-markdown -p app –render-toc > ../../docs/api/analysis.md
@echo “$(GREEN)✓ API documentation built$(NC)”

# ============================================================================

# Monitoring

# ============================================================================

logs-grafana: ## Open Grafana in browser
@echo “$(BLUE)Opening Grafana…$(NC)”
open http://localhost:3001 || xdg-open http://localhost:3001 || echo “Open http://localhost:3001 in your browser”

logs-prometheus: ## Open Prometheus in browser
@echo “$(BLUE)Opening Prometheus…$(NC)”
open http://localhost:9090 || xdg-open http://localhost:9090 || echo “Open http://localhost:9090 in your browser”

stats: ## Show service statistics
@echo “$(BLUE)Service Statistics:$(NC)”
@echo “”
docker stats –no-stream –format “table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}”

health: ## Check health of all services
@echo “$(BLUE)Checking service health…$(NC)”
@echo “”
@docker-compose -f infra/docker/docker-compose.yml ps –format json | jq -r ‘.[] | “(.Name): (.State)”’

# ============================================================================

# Development Utilities

# ============================================================================

install-deps: ## Install all dependencies
@echo “$(BLUE)Installing dependencies…$(NC)”
@echo “$(YELLOW)Backend (Python):$(NC)”
cd services/instruments && pip install -r requirements.txt -r requirements-dev.txt
cd services/analysis && pip install -r requirements.txt -r requirements-dev.txt
cd packages/common && pip install -e .
@echo “$(YELLOW)Frontend (Node.js):$(NC)”
cd apps/web && pnpm install
@echo “$(GREEN)✓ Dependencies installed$(NC)”

setup: ## Initial setup (first-time only)
@echo “$(BLUE)Setting up SemiconductorLab Platform…$(NC)”
@echo “”
@echo “$(YELLOW)Step 1/5: Installing dependencies…$(NC)”
$(MAKE) install-deps
@echo “”
@echo “$(YELLOW)Step 2/5: Starting services…$(NC)”
$(MAKE) dev-up
@echo “”
@echo “$(YELLOW)Step 3/5: Waiting for services…$(NC)”
sleep 15
@echo “”
@echo “$(YELLOW)Step 4/5: Running migrations…$(NC)”
$(MAKE) migrate
@echo “”
@echo “$(YELLOW)Step 5/5: Seeding database…$(NC)”
$(MAKE) seed-db
@echo “”
@echo “$(GREEN)✓ Setup complete!$(NC)”
@echo “”
@echo “$(BLUE)Next steps:$(NC)”
@echo “  1. Access web UI: http://localhost:3000”
@echo “  2. View API docs: http://localhost:8000/docs”
@echo “  3. Generate test data: make generate-test-data”
@echo “  4. Run tests: make test”
@echo “”

version: ## Show version information
@echo “$(BLUE)SemiconductorLab Platform$(NC)”
@echo “Version: 2.0.0”
@echo “Session: S1-S2 Complete”
@echo “”
@echo “Components:”
@echo “  - Database: PostgreSQL 15 + TimescaleDB”
@echo “  - Backend: FastAPI + Python 3.11”
@echo “  - Frontend: Next.js 14”
@echo “  - Message Broker: NATS 2.10”
@echo “  - Storage: MinIO”
@echo “”