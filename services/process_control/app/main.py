"""Main FastAPI application for Process Control service."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting Process Control service...")

    # Initialize hardware (if in production mode)
    # await startup_hardware()

    logger.info("Process Control service startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Process Control service...")

    # Cleanup hardware
    # await shutdown_hardware()

    logger.info("Process Control service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="SPECTRA-Lab Process Control Service",
    description="RTP and Ion Implantation control system with SPC and Virtual Metrology",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# New API v2 routers (with RBAC, Celery, WebSocket)
from app.routers.ion import router as ion_router
from app.routers.rtp import router as rtp_router
from app.routers.jobs import router as jobs_router
from app.routers.websocket import router as websocket_router

# Legacy routers (keep for backward compatibility)
try:
    from app.api.endpoints import spc_router, vm_router
    from app.api.safety_endpoints import safety_router
    app.include_router(spc_router)
    app.include_router(vm_router)
    app.include_router(safety_router)
except ImportError:
    logger.warning("Legacy API routers not found, skipping")

# Register new routers
app.include_router(ion_router)
app.include_router(rtp_router)
app.include_router(jobs_router)
app.include_router(websocket_router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "process-control",
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SPECTRA-Lab Process Control Service",
        "modules": ["Ion Implantation", "RTP", "SPC", "Virtual Metrology"],
        "documentation": "/docs",
        "status": "operational"
    }
