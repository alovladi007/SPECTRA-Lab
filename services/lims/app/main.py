"""
SPECTRA-Lab LIMS API - Main application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import samples_router, recipes_router, sops_router, eln_router

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

# Include routers
app.include_router(samples_router)
app.include_router(recipes_router)
app.include_router(sops_router)
app.include_router(eln_router)


@app.get("/")
async def root():
    return {
        "service": "lims",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/api/samples",
            "/api/recipes",
            "/api/sops",
            "/api/eln"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "lims"}
