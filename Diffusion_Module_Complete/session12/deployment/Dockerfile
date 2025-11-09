# Multi-stage Dockerfile for Diffusion Module
# Production-ready container with optimized builds

# ============================================================================
# Stage 1: Base Python Image with System Dependencies
# ============================================================================
FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

# ============================================================================
# Stage 2: Dependencies
# ============================================================================
FROM base AS dependencies

WORKDIR /tmp

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-warn-script-location \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pandas==2.0.0 \
    matplotlib==3.7.1 \
    scikit-learn==1.2.2 \
    xgboost==1.7.5 \
    pydantic==2.0.0 \
    fastapi==0.100.0 \
    uvicorn[standard]==0.23.0 \
    python-multipart==0.0.6 \
    pyarrow==12.0.0 \
    emcee==3.1.4 \
    numba==0.57.0

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM base AS application

# Copy dependencies from builder
COPY --from=dependencies /root/.local /home/appuser/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser integrated/ /app/integrated/
COPY --chown=appuser:appuser session11/spectra/ /app/session11/spectra/
COPY --chown=appuser:appuser session10/scripts/ /app/scripts/

# Switch to app user
USER appuser

# Add local bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["uvicorn", "integrated.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Stage 4: Development (Optional)
# ============================================================================
FROM application AS development

USER root

# Install development tools
RUN pip install --no-cache-dir \
    pytest==7.3.1 \
    pytest-cov==4.1.0 \
    black==23.3.0 \
    mypy==1.3.0 \
    ruff==0.0.272 \
    ipython==8.14.0 \
    jupyter==1.0.0

USER appuser

CMD ["uvicorn", "integrated.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
