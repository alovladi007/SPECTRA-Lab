"""
Simulation API Schemas

Pydantic models for simulation API requests and responses.
This file will be populated with the actual schemas.py from the Diffusion Module Skeleton Package.

Expected schemas:
- DiffusionRequest: Parameters for diffusion simulation
- DiffusionResponse: Simulation results and profiles
- OxidationRequest: Parameters for oxidation simulation
- OxidationResponse: Oxide thickness and growth data
- CalibrationRequest: Parameters for model calibration
- CalibrationResponse: Calibrated model parameters
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


# ============================================================================
# DIFFUSION SIMULATION SCHEMAS
# ============================================================================

class DiffusionRequest(BaseModel):
    """Request model for diffusion simulation"""
    temperature: float = Field(..., description="Temperature in Celsius", ge=500, le=1400)
    time: float = Field(..., description="Diffusion time in minutes", gt=0)
    dopant: str = Field("boron", description="Dopant type: boron, phosphorus, arsenic")
    initial_concentration: float = Field(1e20, description="Initial dopant concentration (atoms/cm³)")
    depth: float = Field(1.0, description="Simulation depth in micrometers", gt=0)
    grid_points: int = Field(100, description="Number of spatial grid points", ge=10, le=1000)
    model: str = Field("fick", description="Diffusion model: fick, massoud, erfc")

    class Config:
        schema_extra = {
            "example": {
                "temperature": 1000.0,
                "time": 60.0,
                "dopant": "boron",
                "initial_concentration": 1e20,
                "depth": 1.0,
                "grid_points": 100,
                "model": "fick"
            }
        }


class DiffusionProfile(BaseModel):
    """Dopant concentration profile"""
    depth: List[float] = Field(..., description="Depth points in micrometers")
    concentration: List[float] = Field(..., description="Concentration at each depth (atoms/cm³)")


class DiffusionResponse(BaseModel):
    """Response model for diffusion simulation"""
    simulation_id: str = Field(..., description="Unique simulation ID")
    status: str = Field("completed", description="Simulation status")
    profile: DiffusionProfile = Field(..., description="Concentration vs depth profile")
    junction_depth: Optional[float] = Field(None, description="Junction depth in micrometers")
    sheet_resistance: Optional[float] = Field(None, description="Sheet resistance in ohms/square")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


# ============================================================================
# OXIDATION SIMULATION SCHEMAS
# ============================================================================

class OxidationRequest(BaseModel):
    """Request model for oxidation simulation"""
    temperature: float = Field(..., description="Temperature in Celsius", ge=700, le=1200)
    time: float = Field(..., description="Oxidation time in minutes", gt=0)
    ambient: str = Field("dry", description="Ambient type: dry or wet")
    pressure: float = Field(1.0, description="Pressure in atmospheres", gt=0)
    initial_oxide_thickness: float = Field(0.0, description="Initial oxide thickness in nm", ge=0)

    class Config:
        schema_extra = {
            "example": {
                "temperature": 1000.0,
                "time": 120.0,
                "ambient": "dry",
                "pressure": 1.0,
                "initial_oxide_thickness": 0.0
            }
        }


class OxidationResponse(BaseModel):
    """Response model for oxidation simulation"""
    simulation_id: str = Field(..., description="Unique simulation ID")
    status: str = Field("completed", description="Simulation status")
    final_thickness: float = Field(..., description="Final oxide thickness in nm")
    growth_rate: float = Field(..., description="Average growth rate in nm/min")
    time_points: List[float] = Field(..., description="Time points in minutes")
    thickness_profile: List[float] = Field(..., description="Thickness at each time point")
    deal_grove_params: Optional[Dict[str, float]] = Field(None, description="Deal-Grove model parameters")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


# ============================================================================
# CALIBRATION SCHEMAS
# ============================================================================

class CalibrationDataPoint(BaseModel):
    """Single calibration data point"""
    temperature: float = Field(..., description="Temperature in Celsius")
    time: float = Field(..., description="Process time in minutes")
    measured_value: float = Field(..., description="Measured output value")
    process_params: Optional[Dict[str, Any]] = Field(None, description="Additional process parameters")


class CalibrationRequest(BaseModel):
    """Request model for model calibration"""
    model_type: str = Field(..., description="Model type to calibrate: diffusion or oxidation")
    calibration_data: List[CalibrationDataPoint] = Field(..., description="Calibration data points")
    optimization_method: str = Field("least_squares", description="Optimization method")

    class Config:
        schema_extra = {
            "example": {
                "model_type": "oxidation",
                "calibration_data": [
                    {
                        "temperature": 1000.0,
                        "time": 60.0,
                        "measured_value": 25.0
                    },
                    {
                        "temperature": 1100.0,
                        "time": 60.0,
                        "measured_value": 45.0
                    }
                ],
                "optimization_method": "least_squares"
            }
        }


class CalibrationResponse(BaseModel):
    """Response model for model calibration"""
    calibration_id: str = Field(..., description="Unique calibration ID")
    status: str = Field("completed", description="Calibration status")
    calibrated_params: Dict[str, float] = Field(..., description="Calibrated model parameters")
    rmse: float = Field(..., description="Root mean square error")
    r_squared: float = Field(..., description="R² coefficient of determination")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


# ============================================================================
# GENERIC RESPONSE SCHEMAS
# ============================================================================

class SimulationJob(BaseModel):
    """Simulation job status"""
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status: pending, running, completed, failed")
    progress: float = Field(0.0, description="Progress percentage (0-100)")
    message: Optional[str] = Field(None, description="Status message")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Export all schemas
__all__ = [
    "DiffusionRequest",
    "DiffusionProfile",
    "DiffusionResponse",
    "OxidationRequest",
    "OxidationResponse",
    "CalibrationDataPoint",
    "CalibrationRequest",
    "CalibrationResponse",
    "SimulationJob",
    "ErrorResponse"
]
