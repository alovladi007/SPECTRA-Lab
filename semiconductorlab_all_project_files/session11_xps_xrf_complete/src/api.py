"""
Session 11: XPS/XRF Analysis API
FastAPI backend service for chemical analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import json
import io
import os
from datetime import datetime
import logging

# Import the analyzer module
from chemical_analyzer import (
    XPSAnalyzer, XRFAnalyzer, ChemicalSimulator, 
    ElementDatabase, XRaySource, PeakShape
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="XPS/XRF Analysis API",
    description="Chemical surface and elemental analysis service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class XPSParameters(BaseModel):
    """XPS acquisition parameters"""
    source: str = Field(default="Al Kα", description="X-ray source")
    pass_energy: float = Field(default=20.0, description="Pass energy in eV")
    dwell_time: float = Field(default=50.0, description="Dwell time in ms")
    scans: int = Field(default=10, description="Number of scans")
    start_energy: float = Field(default=0.0, description="Start binding energy")
    end_energy: float = Field(default=1200.0, description="End binding energy")
    step_size: float = Field(default=0.1, description="Energy step size")

class XRFParameters(BaseModel):
    """XRF acquisition parameters"""
    excitation_energy: float = Field(default=50.0, description="Excitation energy in keV")
    measurement_time: float = Field(default=300.0, description="Measurement time in seconds")
    atmosphere: str = Field(default="air", description="Measurement atmosphere")
    detector_type: str = Field(default="Si", description="Detector type")

class PeakFitRequest(BaseModel):
    """Peak fitting request"""
    binding_energy: List[float] = Field(..., description="Binding energy array")
    intensity: List[float] = Field(..., description="Intensity array")
    shape: str = Field(default="Voigt", description="Peak shape")
    background_type: str = Field(default="shirley", description="Background type")

class QuantificationResult(BaseModel):
    """Quantification result"""
    element: str
    concentration: float
    error: float
    orbital: Optional[str] = None
    line: Optional[str] = None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "XPS/XRF Analysis API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/chemical/health",
            "xps": "/api/chemical/xps/analyze",
            "xrf": "/api/chemical/xrf/analyze"
        }
    }

@app.post("/api/chemical/xps/analyze")
async def analyze_xps_spectrum(
    file: UploadFile = File(...),
    params: Optional[XPSParameters] = None
):
    """
    Analyze XPS spectrum
    
    Upload a text file with two columns: binding energy and intensity
    """
    try:
        # Set default parameters if not provided
        if params is None:
            params = XPSParameters()
        
        # Read uploaded file
        content = await file.read()
        lines = content.decode().strip().split('\n')
        
        # Parse data (skip header if present)
        data_lines = [l for l in lines if not l.startswith('#')]
        data = []
        for line in data_lines:
            if line.strip():
                values = line.split()
                if len(values) >= 2:
                    try:
                        data.append([float(values[0]), float(values[1])])
                    except ValueError:
                        continue
        
        if not data:
            raise ValueError("No valid data found in file")
        
        data = np.array(data)
        be = data[:, 0]
        intensity = data[:, 1]
        
        # Create analyzer
        source_map = {
            "Al Kα": XRaySource.AL_KA,
            "Mg Kα": XRaySource.MG_KA,
            "Monochromatic Al": XRaySource.MONOCHROMATIC_AL
        }
        source = source_map.get(params.source, XRaySource.AL_KA)
        
        analyzer = XPSAnalyzer(source=source)
        
        # Process spectrum
        be_proc, int_proc = analyzer.process_spectrum(be, intensity)
        
        # Find peaks
        peaks = analyzer.find_peaks(be_proc, int_proc)
        
        # Calculate background
        background = analyzer.shirley_background(be_proc, int_proc)
        
        # Simple quantification (if peaks found)
        composition = {}
        if peaks:
            # Create simplified peak list for quantification
            from chemical_analyzer import XPSPeak, PeakShape
            xps_peaks = []
            for peak in peaks[:5]:  # Take up to 5 peaks
                if peak.get('element'):
                    elem = peak['element'].split()[0] if ' ' in peak['element'] else peak['element']
                    xps_peaks.append(XPSPeak(
                        position=peak['position'],
                        area=peak['intensity'] * 100,  # Simplified area
                        fwhm=1.5,
                        shape=PeakShape.VOIGT,
                        orbital='1s',
                        element=elem
                    ))
            
            if xps_peaks:
                composition = analyzer.quantification(xps_peaks)
        
        return {
            "status": "success",
            "binding_energy": be_proc.tolist(),
            "intensity": int_proc.tolist(),
            "background": background.tolist(),
            "peaks": peaks,
            "composition": composition,
            "parameters": params.dict()
        }
        
    except Exception as e:
        logger.error(f"XPS analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chemical/xps/fit_peak")
async def fit_xps_peak(request: PeakFitRequest):
    """Fit a single XPS peak"""
    try:
        analyzer = XPSAnalyzer()
        
        be = np.array(request.binding_energy)
        intensity = np.array(request.intensity)
        
        # Map shape string to enum
        shape_map = {
            "GAUSSIAN": PeakShape.GAUSSIAN,
            "LORENTZIAN": PeakShape.LORENTZIAN,
            "VOIGT": PeakShape.VOIGT,
            "DONIACH_SUNJIC": PeakShape.DONIACH_SUNJIC,
            "PSEUDO_VOIGT": PeakShape.PSEUDO_VOIGT
        }
        shape = shape_map.get(request.shape.upper(), PeakShape.VOIGT)
        
        # Fit peak
        result = analyzer.fit_peak(
            be, intensity, 
            shape=shape,
            background_type=request.background_type
        )
        
        if result['success']:
            return {
                "status": "success",
                "position": result['position'],
                "amplitude": result['amplitude'],
                "fwhm": result['fwhm'],
                "area": result['area'],
                "r_squared": result['r_squared'],
                "fitted_curve": result['fitted_curve'].tolist(),
                "background": result['background'].tolist()
            }
        else:
            raise ValueError(result.get('error', 'Fitting failed'))
            
    except Exception as e:
        logger.error(f"Peak fitting error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chemical/xrf/analyze")
async def analyze_xrf_spectrum(
    file: UploadFile = File(...),
    params: Optional[XRFParameters] = None
):
    """
    Analyze XRF spectrum
    
    Upload a text file with two columns: energy (keV) and counts
    """
    try:
        # Set default parameters if not provided
        if params is None:
            params = XRFParameters()
        
        # Read uploaded file
        content = await file.read()
        lines = content.decode().strip().split('\n')
        
        # Parse data
        data_lines = [l for l in lines if not l.startswith('#')]
        data = []
        for line in data_lines:
            if line.strip():
                values = line.split()
                if len(values) >= 2:
                    try:
                        data.append([float(values[0]), float(values[1])])
                    except ValueError:
                        continue
        
        if not data:
            raise ValueError("No valid data found in file")
        
        data = np.array(data)
        energy = data[:, 0]
        counts = data[:, 1]
        
        # Create analyzer
        analyzer = XRFAnalyzer(excitation_energy=params.excitation_energy)
        
        # Process spectrum
        energy_proc, counts_proc = analyzer.process_spectrum(energy, counts)
        
        # Find peaks
        peaks = analyzer.find_peaks(energy_proc, counts_proc)
        
        # Quantification
        composition = analyzer.standardless_quantification(energy_proc, counts_proc)
        
        # Detection limits
        mdl = analyzer.detection_limits(
            energy_proc, counts_proc,
            measurement_time=params.measurement_time
        )
        
        return {
            "status": "success",
            "energy": energy_proc.tolist(),
            "counts": counts_proc.tolist(),
            "peaks": peaks,
            "composition": composition,
            "detection_limits": mdl,
            "parameters": params.dict()
        }
        
    except Exception as e:
        logger.error(f"XRF analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chemical/simulate/xps")
async def simulate_xps(composition: Dict[str, float]):
    """Generate simulated XPS spectrum"""
    try:
        simulator = ChemicalSimulator()
        be, intensity = simulator.generate_xps_spectrum(composition)
        
        return {
            "status": "success",
            "binding_energy": be.tolist(),
            "intensity": intensity.tolist(),
            "composition": composition
        }
        
    except Exception as e:
        logger.error(f"XPS simulation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chemical/simulate/xrf")
async def simulate_xrf(composition: Dict[str, float]):
    """Generate simulated XRF spectrum"""
    try:
        simulator = ChemicalSimulator()
        energy, counts = simulator.generate_xrf_spectrum(composition)
        
        return {
            "status": "success",
            "energy": energy.tolist(),
            "counts": counts.tolist(),
            "composition": composition
        }
        
    except Exception as e:
        logger.error(f"XRF simulation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/chemical/elements")
async def get_elements():
    """Get available elements database"""
    try:
        db = ElementDatabase()
        elements = []
        
        for symbol, element in db.elements.items():
            elements.append({
                "symbol": symbol,
                "name": element.name,
                "atomic_number": element.atomic_number,
                "atomic_mass": element.atomic_mass,
                "xps_peaks": list(element.xps_peaks.keys()),
                "xrf_lines": list(element.xrf_lines.keys()),
                "fluorescence_yield": element.fluorescence_yield
            })
        
        return {
            "status": "success",
            "count": len(elements),
            "elements": elements
        }
        
    except Exception as e:
        logger.error(f"Element database error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chemical/health")
async def health_check():
    """Check service health"""
    try:
        # Test that we can create analyzers
        xps = XPSAnalyzer()
        xrf = XRFAnalyzer()
        
        return {
            "status": "healthy",
            "service": "XPS/XRF Analysis",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                "xps": True,
                "xrf": True,
                "simulation": True,
                "elements_db": True
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "type": "ValueError"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("XPS/XRF Analysis API starting up...")
    logger.info("Service ready at http://localhost:8011")
    logger.info("API docs available at http://localhost:8011/docs")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("XPS/XRF Analysis API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011, reload=True)
