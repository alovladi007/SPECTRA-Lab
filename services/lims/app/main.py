"""
SPECTRA-Lab LIMS API - Main application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# For now, only import auth router (others require database setup)
from .api.auth import router as auth_router

app = FastAPI(
    title="SPECTRA-Lab LIMS API",
    version="1.0.0",
    description="Laboratory Information Management System API for SPECTRA-Lab Platform"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth router
app.include_router(auth_router)

# Note: Other routers (samples, recipes, sops, eln) are temporarily disabled until database is initialized
# They will be enabled once database connectivity is established


@app.get("/")
async def root():
    return {
        "service": "lims",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/api/auth/login",
            "/health"
        ],
        "note": "Database-dependent endpoints temporarily disabled pending DB initialization"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "lims"}
