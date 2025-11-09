"""
Simulation API Routers

FastAPI endpoints for diffusion and oxidation simulations.
This file will be populated with the actual routers.py from the Diffusion Module Skeleton Package.

Expected endpoints:
- POST /simulation/diffusion - Run diffusion simulation
- POST /simulation/oxidation - Run oxidation simulation
- POST /simulation/calibrate - Calibrate model parameters
- GET /simulation/jobs/{job_id} - Get job status
- GET /simulation/results/{simulation_id} - Get simulation results
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
import uuid
import logging
from datetime import datetime
import numpy as np

from .schemas import (
    DiffusionRequest,
    DiffusionResponse,
    DiffusionProfile,
    OxidationRequest,
    OxidationResponse,
    CalibrationRequest,
    CalibrationResponse,
    SimulationJob,
    ErrorResponse
)

# Import diffusion simulation functions
from app.simulation.diffusion import (
    constant_source_profile,
    limited_source_profile,
    junction_depth as calc_junction_depth,
    sheet_resistance_estimate,
    quick_profile_constant_source,
)

# Import database functions (with graceful fallback)
try:
    from app.simulation.database import (
        save_simulation,
        get_simulation,
        health_check as db_health_check
    )
    DATABASE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Database integration enabled")
except Exception as e:
    DATABASE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Database not available, using in-memory storage: {e}")

router = APIRouter(prefix="/simulation", tags=["Simulation"])

# In-memory storage (fallback when database unavailable)
jobs_db: Dict[str, Dict] = {}
results_db: Dict[str, Dict] = {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_simulation_result(simulation_id: str, simulation_type: str, parameters: dict, results: dict, execution_time_ms: int = None):
    """
    Save simulation result to database (if available) or in-memory storage

    Args:
        simulation_id: Unique simulation identifier
        simulation_type: Type of simulation (diffusion, oxidation, etc.)
        parameters: Input parameters
        results: Simulation results
        execution_time_ms: Execution time in milliseconds
    """
    if DATABASE_AVAILABLE:
        try:
            save_simulation({
                "simulation_id": simulation_id,
                "simulation_type": simulation_type,
                "parameters": parameters,
                "results": results,
                "status": "completed",
                "execution_time_ms": execution_time_ms,
                "module_version": "1.12.0",
                "created_at": datetime.utcnow()
            })
            logger.debug(f"Simulation {simulation_id} saved to database")
        except Exception as e:
            logger.warning(f"Failed to save to database, using memory: {e}")
            results_db[simulation_id] = {**results, "parameters": parameters}
    else:
        results_db[simulation_id] = {**results, "parameters": parameters}


def get_simulation_result(simulation_id: str) -> Optional[dict]:
    """Get simulation result from database or in-memory storage"""
    if DATABASE_AVAILABLE:
        try:
            result = get_simulation(simulation_id)
            if result:
                return result
        except Exception as e:
            logger.warning(f"Failed to get from database, checking memory: {e}")

    return results_db.get(simulation_id)


# ============================================================================
# DIFFUSION SIMULATION ENDPOINTS
# ============================================================================

@router.post("/diffusion", response_model=DiffusionResponse)
async def run_diffusion_simulation(request: DiffusionRequest):
    """
    Run a diffusion simulation using ERFC analytical solutions

    Supports:
    - Constant source diffusion (surface concentration held constant)
    - Limited source diffusion (Gaussian from fixed dose)
    - Common dopants: boron, phosphorus, arsenic

    Uses Session 2 implementation - Production ready!
    """
    import time
    try:
        # Generate unique ID and start timer
        simulation_id = str(uuid.uuid4())
        start_time = time.time()

        # Use quick helper function for constant-source diffusion (default)
        if request.model in ["fick", "erfc"]:
            # Create depth array
            x = np.linspace(0, request.depth, request.grid_points)

            # Convert time from minutes to seconds
            time_seconds = request.time * 60

            # Get dopant parameters
            dopant_params = {
                "boron": (0.76, 3.46),
                "phosphorus": (3.85, 3.66),
                "arsenic": (0.066, 3.44)
            }

            D0, Ea = dopant_params.get(request.dopant, (0.76, 3.46))

            # Run simulation
            C = constant_source_profile(
                x=x,
                t=time_seconds,
                T=request.temperature,
                D0=D0,
                Ea=Ea,
                Cs=request.initial_concentration,
                NA0=1e15  # Background doping
            )

            # Calculate junction depth
            try:
                xj = calc_junction_depth(C, x, 1e15, method="linear")
            except ValueError:
                xj = None

            # Calculate sheet resistance (assume n-type for P, As; p-type for B)
            dopant_type = "p" if request.dopant == "boron" else "n"
            Rs = sheet_resistance_estimate(C, x, dopant_type=dopant_type)

            # Build response
            response = DiffusionResponse(
                simulation_id=simulation_id,
                status="completed",
                profile=DiffusionProfile(
                    depth=x.tolist(),
                    concentration=C.tolist()
                ),
                junction_depth=float(xj) if xj is not None else None,
                sheet_resistance=float(Rs),
                metadata={
                    "model": "erfc (constant source)",
                    "temperature": request.temperature,
                    "time": request.time,
                    "dopant": request.dopant,
                    "D0": D0,
                    "Ea": Ea,
                    "implementation": "Session 2 - Production Ready"
                }
            )
        else:
            raise ValueError(f"Model '{request.model}' not yet implemented. Use 'fick' or 'erfc'.")

        # Store result (database or in-memory)
        execution_time_ms = int((time.time() - start_time) * 1000)
        save_simulation_result(
            simulation_id=simulation_id,
            simulation_type="diffusion",
            parameters=request.dict(),
            results=response.dict(),
            execution_time_ms=execution_time_ms
        )

        logger.info(f"Diffusion simulation {simulation_id} completed in {execution_time_ms}ms using {request.model} model")
        return response

    except Exception as e:
        logger.error(f"Diffusion simulation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diffusion/{simulation_id}", response_model=DiffusionResponse)
async def get_diffusion_result(simulation_id: str):
    """Get diffusion simulation result by ID"""
    result = get_simulation_result(simulation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")

    return result


# ============================================================================
# OXIDATION SIMULATION ENDPOINTS
# ============================================================================

@router.post("/oxidation", response_model=OxidationResponse)
async def run_oxidation_simulation(request: OxidationRequest):
    """
    Run an oxidation simulation using Deal-Grove model

    This endpoint will call the actual oxidation simulation module (deal_grove.py).

    **Implementation will be completed when oxidation modules are integrated.**
    """
    import time
    try:
        # Generate unique ID and start timer
        simulation_id = str(uuid.uuid4())
        start_time = time.time()

        # Placeholder response
        # TODO: Replace with actual Deal-Grove simulation call
        time_points = [i * request.time / 100 for i in range(101)]
        thickness_profile = [
            request.initial_oxide_thickness + (i * 0.5) for i in range(101)
        ]

        response = OxidationResponse(
            simulation_id=simulation_id,
            status="completed",
            final_thickness=thickness_profile[-1],
            growth_rate=thickness_profile[-1] / request.time,
            time_points=time_points,
            thickness_profile=thickness_profile,
            deal_grove_params={
                "B": 0.1,  # Linear rate constant
                "B_over_A": 0.5,  # Parabolic rate constant / linear rate constant
                "tau": 0.0  # Time offset
            }
        )

        # Store result (database or in-memory)
        execution_time_ms = int((time.time() - start_time) * 1000)
        save_simulation_result(
            simulation_id=simulation_id,
            simulation_type="oxidation",
            parameters=request.dict(),
            results=response.dict(),
            execution_time_ms=execution_time_ms
        )

        logger.info(f"Oxidation simulation {simulation_id} completed in {execution_time_ms}ms")
        return response

    except Exception as e:
        logger.error(f"Oxidation simulation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/oxidation/{simulation_id}", response_model=OxidationResponse)
async def get_oxidation_result(simulation_id: str):
    """Get oxidation simulation result by ID"""
    result = get_simulation_result(simulation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")

    return result


# ============================================================================
# MODEL CALIBRATION ENDPOINTS
# ============================================================================

@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_model(request: CalibrationRequest):
    """
    Calibrate simulation model parameters from experimental data

    This endpoint will call the actual calibration module (calibrate.py).

    **Implementation will be completed when calibration modules are integrated.**
    """
    try:
        # Generate unique ID
        calibration_id = str(uuid.uuid4())

        # Placeholder response
        # TODO: Replace with actual calibration call
        response = CalibrationResponse(
            calibration_id=calibration_id,
            status="completed",
            calibrated_params={
                "B": 0.12,
                "B_over_A": 0.45,
                "activation_energy": 1.2
            },
            rmse=2.5,
            r_squared=0.98
        )

        # Store result
        results_db[calibration_id] = response.dict()

        logger.info(f"Model calibration {calibration_id} completed")
        return response

    except Exception as e:
        logger.error(f"Model calibration failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# JOB MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/jobs/{job_id}", response_model=SimulationJob)
async def get_job_status(job_id: str):
    """Get simulation job status"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs_db[job_id]


@router.get("/jobs", response_model=List[SimulationJob])
async def list_jobs(limit: int = 50):
    """List all simulation jobs"""
    return list(jobs_db.values())[:limit]


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check for simulation service"""
    db_status = "disconnected"
    if DATABASE_AVAILABLE:
        try:
            if db_health_check():
                db_status = "connected"
        except:
            db_status = "error"

    return {
        "status": "healthy",
        "service": "simulation",
        "database_status": db_status,
        "storage_mode": "database" if (DATABASE_AVAILABLE and db_status == "connected") else "memory",
        "simulations_completed": len(results_db),
        "active_jobs": len([j for j in jobs_db.values() if j.get("status") == "running"])
    }


@router.delete("/clear")
async def clear_all_data():
    """Clear all simulation data (development only)"""
    global jobs_db, results_db
    jobs_db = {}
    results_db = {}
    return {"message": "All simulation data cleared"}


# Export router
__all__ = ["router"]
