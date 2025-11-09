"""
Simulation API Endpoints

FastAPI routers for simulation services:
- Diffusion simulation endpoints
- Oxidation simulation endpoints
- Process calibration endpoints
"""

from .routers import router as simulation_router

__all__ = ["simulation_router"]
