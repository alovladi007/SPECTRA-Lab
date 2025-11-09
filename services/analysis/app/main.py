"""
Analysis Service - Port 8001
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import ML routers
from app.api.v1 import automl_router, explainability_router, ab_testing_router, monitoring_router
from app.api.v1.simulation import simulation_router

# Import runs and calibrations routers
from app.api import runs_router, calibrations_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="SPECTRA-Lab Analysis API",
    version="1.0.0",
    description="Analysis and ML services for SPECTRA-Lab platform"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3012"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register ML routers
app.include_router(automl_router, prefix="/api/v1")
app.include_router(explainability_router, prefix="/api/v1")
app.include_router(ab_testing_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")

# Register simulation routers
app.include_router(simulation_router, prefix="/api/v1")

# Register runs and calibrations routers
app.include_router(runs_router)
app.include_router(calibrations_router)

@app.get("/")
async def root():
    return {
        "service": "analysis",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "automl": "/api/v1/automl",
            "explainability": "/api/v1/explainability",
            "ab_testing": "/api/v1/ab-testing",
            "monitoring": "/api/v1/monitoring",
            "simulation": "/api/v1/simulation",
            "runs": "/api/v1/runs",
            "calibrations": "/api/v1/calibrations"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "analysis"}
