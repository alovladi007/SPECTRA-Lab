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
	@echo "$(BLUE)SemiconductorLab Platform - Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# Development Environment
# ============================================================================

dev-up: ## Start all development services
	@echo "$(BLUE)Starting SemiconductorLab development environment...$(NC)"
	docker-compose -f infra/docker/docker-compose.yml up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo ""
	@echo "Access points:"
	@echo "  - Web UI:        http://localhost:3000"
	@echo "  - API Docs:      http://localhost:8000/docs"
	@echo "  - Analysis API:  http://localhost:8001/docs"
	@echo "  - Grafana:       http://localhost:3001 (admin/admin)"
	@echo "  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
	@echo "  - Prometheus:    http://localhost:9090"
	@echo ""

dev-down: ## Stop all development services
	@echo "$(YELLOW)Stopping SemiconductorLab development environment...$(NC)"
	docker-compose -f infra/docker/docker-compose.yml down
	@echo "$(GREEN)✓ Services stopped$(NC)"

dev-logs: ## Tail logs from all services
	docker-compose -f infra/docker/docker-compose.yml logs -f

dev-logs-service: ## Tail logs from specific service (usage: make dev-logs-service SERVICE=instruments)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		echo "Usage: make dev-logs-service SERVICE=instruments"; \
		exit 1; \
	fi
	docker-compose -f infra/docker/docker-compose.yml logs -f $(SERVICE)

dev-reset: ## Reset entire development environment (WARNING: deletes all data)
	@echo "$(RED)WARNING: This will delete all data. Press Ctrl+C to cancel, Enter to continue.$(NC)"
	@read confirm
	@echo "$(YELLOW)Resetting development environment...$(NC)"
	docker-compose -f infra/docker/docker-compose.yml down -v
	@echo "$(GREEN)✓ Volumes deleted$(NC)"
	$(MAKE) dev-up
	@echo "$(BLUE)Waiting for services to be ready...$(NC)"
	sleep 10
	$(MAKE) migrate
	$(MAKE) seed-db
	@echo "$(GREEN)✓ Development environment reset complete$(NC)"

dev-shell: ## Open shell in a service container (usage: make dev-shell SERVICE=instruments)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		echo "Usage: make dev-shell SERVICE=instruments"; \
		exit 1; \
	fi
	docker-compose -f infra/docker/docker-compose.yml exec $(SERVICE) /bin/bash

# ============================================================================
# Database Management
# ============================================================================

migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	docker-compose -f infra/docker/docker-compose.yml exec instruments alembic upgrade head
	@echo "$(GREEN)✓ Migrations applied$(NC)"

migrate-rollback: ## Rollback last migration
	@echo "$(YELLOW)Rolling back last migration...$(NC)"
	docker-compose -f infra/docker/docker-compose.yml exec instruments alembic downgrade -1
	@echo "$(GREEN)✓ Migration rolled back$(NC)"

migrate-create: ## Create new migration (usage: make migrate-create NAME="add_new_table")
	@if [ -z "$(NAME)" ]; then \
		echo "$(RED)Error: NAME parameter required$(NC)"; \
		echo "Usage: make migrate-create NAME=\"add_new_table\""; \
		exit 1; \
	fi
	docker-compose -f infra/docker/docker-compose.yml exec instruments alembic revision -m "$(NAME)"
	@echo "$(GREEN)✓ Migration created$(NC)"

seed-db: ## Seed database with initial data
	@echo "$(BLUE)Seeding database...$(NC)"
	docker-compose -f infra/docker/docker-compose.yml exec instruments python /app/scripts/dev/seed_db.py
	@echo "$(GREEN)✓ Database seeded$(NC)"

db-shell: ## Open PostgreSQL shell
	docker-compose -f infra/docker/docker-compose.yml exec postgres psql -U postgres -d semiconductorlab_dev

db-backup: ## Backup database
	@echo "$(BLUE)Backing up database...$(NC)"
	mkdir -p backups
	docker-compose -f infra/docker/docker-compose.yml exec -T postgres pg_dump -U postgres semiconductorlab_dev > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✓ Database backed up to backups/$(NC)"

db-restore: ## Restore database from backup (usage: make db-restore FILE=backups/backup_20251021_120000.sql)
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Error: FILE parameter required$(NC)"; \
		echo "Usage: make db-restore FILE=backups/backup_20251021_120000.sql"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Restoring database from $(FILE)...$(NC)"
	docker-compose -f infra/docker/docker-compose.yml exec -T postgres psql -U postgres semiconductorlab_dev < $(FILE)
	@echo "$(GREEN)✓ Database restored$(NC)"

# ============================================================================
# Test Data Generation
# ============================================================================

generate-test-data: ## Generate synthetic test data for all methods
	@echo "$(BLUE)Generating test data...$(NC)"
	python scripts/dev/generate_test_data.py
	@echo "$(GREEN)✓ Test data generated in data/test_data/$(NC)"

generate-test-data-method: ## Generate test data for specific method (usage: make generate-test-data-method METHOD=iv_sweep)
	@if [ -z "$(METHOD)" ]; then \
		echo "$(RED)Error: METHOD parameter required$(NC)"; \
		echo "Usage: make generate-test-data-method METHOD=iv_sweep"; \
		exit 1; \
	fi
	python scripts/dev/generate_test_data.py --method $(METHOD)

# ============================================================================
# Testing
# ============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	pytest -v --cov=services --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Tests complete. Coverage report: htmlcov/index.html$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit -v
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	pytest tests/e2e -v
	@echo "$(GREEN)✓ E2E tests complete$(NC)"

test-service: ## Run tests for specific service (usage: make test-service SERVICE=instruments)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		echo "Usage: make test-service SERVICE=instruments"; \
		exit 1; \
	fi
	pytest services/$(SERVICE)/tests -v

test-watch: ## Run tests in watch mode
	ptw -- -v

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linters (ruff, mypy)
	@echo "$(BLUE)Running linters...$(NC)"
	ruff check services packages scripts
	mypy services packages scripts
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code (ruff, black)
	@echo "$(BLUE)Formatting code...$(NC)"
	ruff check --fix services packages scripts
	black services packages scripts
	@echo "$(GREEN)✓ Formatting complete$(NC)"

format-check: ## Check code formatting without changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	ruff check services packages scripts
	black --check services packages scripts

# ============================================================================
# Build & Deploy
# ============================================================================

build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose -f infra/docker/docker-compose.yml build
	@echo "$(GREEN)✓ Images built$(NC)"

build-service: ## Build specific service (usage: make build-service SERVICE=instruments)
	@if [ -z "$(SERVICE)" ]; then \
		echo "$(RED)Error: SERVICE parameter required$(NC)"; \
		echo "Usage: make build-service SERVICE=instruments"; \
		exit 1; \
	fi
	docker-compose -f infra/docker/docker-compose.yml build $(SERVICE)

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	# Add staging deployment commands here
	@echo "$(GREEN)✓ Deployed to staging$(NC)"

deploy-prod: ## Deploy to production (requires confirmation)
	@echo "$(RED)WARNING: This will deploy to PRODUCTION. Type 'yes' to continue:$(NC)"
	@read confirm && [ "$$confirm" = "yes" ] || (echo "Cancelled" && exit 1)
	@echo "$(BLUE)Deploying to production...$(NC)"
	# Add production deployment commands here
	@echo "$(GREEN)✓ Deployed to production$(NC)"

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Remove build artifacts and caches
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean ## Remove all generated files including data and backups
	@echo "$(RED)WARNING: This will delete test data and backups. Press Ctrl+C to cancel, Enter to continue.$(NC)"
	@read confirm
	rm -rf data/test_data/*
	rm -rf backups/*
	@echo "$(GREEN)✓ All generated files removed$(NC)"

# ============================================================================
# Documentation
# ============================================================================

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	cd docs && python -m http.server 8080

docs-build: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	# Add documentation build commands here (e.g., Sphinx, MkDocs)
	@echo "$(GREEN)✓ Documentation built$(NC)"
