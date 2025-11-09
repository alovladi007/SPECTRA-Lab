# Makefile for Diffusion Module
# Production-ready development workflow

.PHONY: help setup clean format lint type test coverage docs api dashboards docker qa benchmark all

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
PIP := $(PYTHON) -m pip

# Directories
SRC_DIRS := integrated session11
TEST_DIRS := integrated/tests session10/tests
SCRIPTS_DIR := session10/scripts
DASHBOARDS_DIR := session11/dashboards

# Docker
DOCKER_IMAGE := diffusion-module
DOCKER_TAG := v12

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Dependencies

setup: ## Create virtual environment and install dependencies
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv venv
	@echo "$(BLUE)Installing dependencies...$(NC)"
	./venv/bin/$(PIP) install --upgrade pip setuptools wheel
	./venv/bin/$(PIP) install -r requirements.txt
	./venv/bin/$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Setup complete! Activate with: source venv/bin/activate$(NC)"

install: ## Install package in development mode
	$(PIP) install -e .

upgrade: ## Upgrade all dependencies
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt

clean: ## Remove build artifacts, cache files, and temp files
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ htmlcov/ .coverage.*
	@echo "$(GREEN)Clean complete!$(NC)"

##@ Code Quality

format: ## Format code with black and ruff
	@echo "$(BLUE)Formatting code with black...$(NC)"
	black $(SRC_DIRS) $(TEST_DIRS)
	@echo "$(BLUE)Sorting imports with ruff...$(NC)"
	ruff --select I --fix $(SRC_DIRS) $(TEST_DIRS)
	@echo "$(GREEN)Format complete!$(NC)"

lint: ## Lint code with ruff
	@echo "$(BLUE)Linting code with ruff...$(NC)"
	ruff $(SRC_DIRS) $(TEST_DIRS)
	@echo "$(GREEN)Lint complete!$(NC)"

type: ## Type check with mypy
	@echo "$(BLUE)Type checking with mypy...$(NC)"
	mypy $(SRC_DIRS) --ignore-missing-imports
	@echo "$(GREEN)Type check complete!$(NC)"

##@ Testing

test: ## Run all tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	pytest $(TEST_DIRS) -v --tb=short

test-fast: ## Run tests without slow markers
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest $(TEST_DIRS) -v -m "not slow"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest $(TEST_DIRS) -v -m "integration"

coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest $(TEST_DIRS) --cov=integrated --cov=session11 --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(NC)"

coverage-xml: ## Generate XML coverage report for CI
	pytest $(TEST_DIRS) --cov=integrated --cov=session11 --cov-report=xml

##@ API & Services

api: ## Run FastAPI development server
	@echo "$(BLUE)Starting FastAPI server...$(NC)"
	uvicorn integrated.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod: ## Run FastAPI production server
	@echo "$(BLUE)Starting FastAPI production server...$(NC)"
	uvicorn integrated.api.main:app --host 0.0.0.0 --port 8000 --workers 4

##@ Dashboards

dashboards: ## Launch all Streamlit dashboards
	@echo "$(BLUE)Launching Streamlit dashboards...$(NC)"
	streamlit run $(DASHBOARDS_DIR)/diffusion_viewer.py --server.port 8501 &
	streamlit run $(DASHBOARDS_DIR)/oxide_planner.py --server.port 8502 &
	streamlit run $(DASHBOARDS_DIR)/spc_monitor.py --server.port 8503 &
	@echo "$(GREEN)Dashboards running on ports 8501-8503$(NC)"

dashboard-diffusion: ## Launch diffusion profile viewer
	streamlit run $(DASHBOARDS_DIR)/diffusion_viewer.py

dashboard-oxidation: ## Launch oxide thickness planner
	streamlit run $(DASHBOARDS_DIR)/oxide_planner.py

dashboard-spc: ## Launch SPC monitor
	streamlit run $(DASHBOARDS_DIR)/spc_monitor.py

##@ Documentation

docs: ## Build documentation with MkDocs
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build
	@echo "$(GREEN)Documentation built: site/index.html$(NC)"

docs-serve: ## Serve documentation locally
	mkdocs serve

##@ Docker

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f session12/deployment/Dockerfile .
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -p 8000:8000 --name diffusion-api $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	cd session12/deployment && docker-compose up -d
	@echo "$(GREEN)Services started$(NC)"

docker-compose-down: ## Stop all services
	@echo "$(YELLOW)Stopping services...$(NC)"
	cd session12/deployment && docker-compose down

docker-compose-logs: ## View docker-compose logs
	cd session12/deployment && docker-compose logs -f

docker-clean: ## Remove Docker containers and images
	docker-compose -f session12/deployment/docker-compose.yml down -v
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) || true

##@ Scripts & Batch Processing

batch-diffusion: ## Run batch diffusion simulation (usage: make batch-diffusion INPUT=runs.csv OUTPUT=results.parquet)
	$(PYTHON) $(SCRIPTS_DIR)/batch_diffusion_sim.py --input $(INPUT) --out $(OUTPUT) --verbose

batch-oxidation: ## Run batch oxidation simulation
	$(PYTHON) $(SCRIPTS_DIR)/batch_oxidation_sim.py --input $(INPUT) --out $(OUTPUT) --verbose

spc-watch: ## Run SPC monitoring (usage: make spc-watch SERIES=kpi.csv REPORT=spc.json)
	$(PYTHON) $(SCRIPTS_DIR)/spc_watch.py --series $(SERIES) --report $(REPORT) --methods all --verbose

##@ Performance & Benchmarks

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(PYTHON) session12/performance/benchmark.py
	@echo "$(GREEN)Benchmark results: session12/performance/benchmark_results.json$(NC)"

profile: ## Profile diffusion solver
	@echo "$(BLUE)Profiling diffusion solver...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats session12/performance/profile_solver.py
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.strip_dirs().sort_stats('cumulative').print_stats(20)"

##@ QA & Validation

qa: ## Run comprehensive QA suite
	@echo "$(BLUE)Running QA suite...$(NC)"
	$(MAKE) lint
	$(MAKE) type
	$(MAKE) coverage
	$(PYTHON) session12/qa/run_regression_tests.py
	@echo "$(GREEN)QA suite complete!$(NC)"

qa-report: ## Generate QA report
	@echo "$(BLUE)Generating QA report...$(NC)"
	$(PYTHON) session12/qa/generate_qa_report.py
	@echo "$(GREEN)QA report: session12/qa/QA_REPORT.md$(NC)"

golden-recipes: ## Run golden recipe validation
	@echo "$(BLUE)Running golden recipe tests...$(NC)"
	$(PYTHON) session12/qa/test_golden_recipes.py

##@ All-in-One

all: clean setup lint type test docs docker-build ## Run complete workflow
	@echo "$(GREEN)Complete workflow finished!$(NC)"

ci: lint type coverage ## Run CI pipeline
	@echo "$(GREEN)CI pipeline complete!$(NC)"

release: qa docker-build ## Prepare for release
	@echo "$(GREEN)Release preparation complete!$(NC)"
