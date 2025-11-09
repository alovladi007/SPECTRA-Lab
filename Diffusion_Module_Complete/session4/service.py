"""
FastAPI service for semiconductor process simulation.

Endpoints:
    POST /oxidation/simulate: Simulate thermal oxidation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np

from core import deal_grove, massoud


app = FastAPI(
    title="Diffusion-Sim API",
    description="Semiconductor process simulation API for thermal oxidation and diffusion",
    version="0.4.0"
)


class OxidationRequest(BaseModel):
    """Request model for oxidation simulation."""
    temperature: float = Field(..., description="Temperature in Celsius", ge=600, le=1400)
    ambient: str = Field(..., description="Oxidation ambient: 'dry' or 'wet'")
    time_points: List[float] = Field(..., description="Time points for simulation (hours)", min_items=1)
    pressure: float = Field(1.0, description="Partial pressure of oxidant (atm)", gt=0)
    initial_thickness: float = Field(0.0, description="Initial oxide thickness (μm)", ge=0)
    use_massoud: bool = Field(True, description="Apply Massoud thin-oxide correction")
    target_thickness: Optional[float] = Field(None, description="Target thickness for inverse calculation (μm)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 1000,
                "ambient": "dry",
                "time_points": [0.5, 1.0, 2.0, 4.0],
                "pressure": 1.0,
                "initial_thickness": 0.0,
                "use_massoud": True,
                "target_thickness": 0.5
            }
        }


class OxidationResponse(BaseModel):
    """Response model for oxidation simulation."""
    time_points: List[float] = Field(..., description="Time points (hours)")
    thickness: List[float] = Field(..., description="Oxide thickness at each time point (μm)")
    thickness_nm: List[float] = Field(..., description="Oxide thickness at each time point (nm)")
    temperature: float = Field(..., description="Temperature (°C)")
    ambient: str = Field(..., description="Oxidation ambient")
    pressure: float = Field(..., description="Partial pressure (atm)")
    rate_constants: dict = Field(..., description="Deal-Grove rate constants")
    massoud_applied: bool = Field(..., description="Whether Massoud correction was applied")
    inverse_solution: Optional[dict] = Field(None, description="Solution to inverse problem if requested")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Diffusion-Sim API",
        "version": "0.4.0",
        "description": "Semiconductor process simulation API",
        "endpoints": {
            "/oxidation/simulate": "POST - Simulate thermal oxidation",
            "/docs": "GET - Interactive API documentation",
            "/redoc": "GET - Alternative API documentation"
        }
    }


@app.post("/oxidation/simulate", response_model=OxidationResponse)
async def simulate_oxidation(request: OxidationRequest):
    """
    Simulate thermal oxidation using Deal-Grove model with optional Massoud correction.
    
    The simulation calculates oxide thickness vs time for given temperature and ambient.
    Optionally solves the inverse problem to find time required for target thickness.
    
    Args:
        request: OxidationRequest with simulation parameters
        
    Returns:
        OxidationResponse with thickness vs time and rate constants
        
    Raises:
        HTTPException: If parameters are invalid
    """
    try:
        # Validate ambient
        if request.ambient.lower() not in ['dry', 'wet']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ambient '{request.ambient}'. Must be 'dry' or 'wet'"
            )
        
        # Get rate constants
        B, B_over_A = deal_grove.get_rate_constants(
            request.temperature,
            request.ambient,
            request.pressure
        )
        A = B / B_over_A
        
        # Convert time points to numpy array
        time_array = np.array(request.time_points)
        
        # Ensure times are non-negative and sorted
        if np.any(time_array < 0):
            raise HTTPException(
                status_code=400,
                detail="Time points must be non-negative"
            )
        
        # Calculate thickness at each time point
        if request.use_massoud:
            thickness_array = massoud.thickness_with_correction(
                time_array,
                request.temperature,
                request.ambient,
                request.pressure,
                request.initial_thickness,
                apply_correction=True
            )
        else:
            thickness_array = deal_grove.thickness_at_time(
                time_array,
                request.temperature,
                request.ambient,
                request.pressure,
                request.initial_thickness
            )
        
        # Convert to lists for JSON serialization
        thickness_um = thickness_array.tolist() if isinstance(thickness_array, np.ndarray) else [thickness_array]
        thickness_nm = [x * 1000 for x in thickness_um]
        
        # Prepare response
        response_data = {
            "time_points": request.time_points,
            "thickness": thickness_um,
            "thickness_nm": thickness_nm,
            "temperature": request.temperature,
            "ambient": request.ambient,
            "pressure": request.pressure,
            "rate_constants": {
                "B": float(B),
                "B_over_A": float(B_over_A),
                "A": float(A),
                "units": {
                    "B": "μm²/hr",
                    "B_over_A": "μm/hr",
                    "A": "μm"
                }
            },
            "massoud_applied": request.use_massoud
        }
        
        # Solve inverse problem if target thickness specified
        if request.target_thickness is not None:
            if request.target_thickness < request.initial_thickness:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target thickness {request.target_thickness} μm < initial thickness {request.initial_thickness} μm"
                )
            
            if request.use_massoud:
                time_required = massoud.time_to_thickness_with_correction(
                    request.target_thickness,
                    request.temperature,
                    request.ambient,
                    request.pressure,
                    request.initial_thickness,
                    apply_correction=True
                )
            else:
                time_required = deal_grove.time_to_thickness(
                    request.target_thickness,
                    request.temperature,
                    request.ambient,
                    request.pressure,
                    request.initial_thickness
                )
            
            # Calculate growth rate at target thickness
            final_rate = deal_grove.growth_rate(
                request.target_thickness,
                request.temperature,
                request.ambient,
                request.pressure
            )
            
            response_data["inverse_solution"] = {
                "target_thickness_um": request.target_thickness,
                "target_thickness_nm": request.target_thickness * 1000,
                "time_required_hr": float(time_required),
                "time_required_min": float(time_required * 60),
                "growth_rate_at_target": float(final_rate),
                "growth_rate_units": "μm/hr"
            }
        
        return OxidationResponse(**response_data)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "diffusion-sim"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
