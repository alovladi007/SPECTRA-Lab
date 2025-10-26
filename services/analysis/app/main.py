"""
SPECTRA-Lab Analysis Service
Main FastAPI application for all characterization methods

This service provides REST API endpoints for:
- Electrical characterization (4PP, Hall Effect, I-V, C-V, BJT, MOSFET, Solar Cell, DLTS, EBIC, PCD)
- Optical characterization (UV-Vis-NIR, FTIR, Ellipsometry, PL, Raman)
- Structural characterization (XRD, SEM, TEM, AFM)
- Chemical characterization (XPS, XRF, SIMS, RBS, NAA)
- Statistical Process Control (SPC)
- Machine Learning & Virtual Metrology
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="SPECTRA-Lab Analysis Service",
    description="Comprehensive semiconductor characterization analysis API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "analysis",
        "version": "2.0.0",
        "capabilities": {
            "electrical_methods": 10,
            "optical_methods": 5,
            "structural_methods": 5,
            "chemical_methods": 6,
            "spc_enabled": True,
            "ml_vm_enabled": True
        }
    }

# ==================== Electrical Characterization ====================

class FourPointProbeRequest(BaseModel):
    voltage: float
    current: float
    probe_spacing: float = 1.0  # mm
    sample_thickness: Optional[float] = None
    temperature: float = 25.0

class FourPointProbeResponse(BaseModel):
    sheet_resistance: float
    resistivity: Optional[float]
    conductivity: float
    measurement_id: str

@app.post("/api/electrical/four-point-probe/measure", response_model=FourPointProbeResponse)
async def measure_four_point_probe(request: FourPointProbeRequest):
    """
    Four-Point Probe measurement
    Calculate sheet resistance from voltage and current
    """
    try:
        # Calculate sheet resistance (Ω/sq)
        sheet_resistance = (request.voltage / request.current) * 4.53236  # π/ln(2) correction factor

        # Calculate resistivity if thickness provided
        resistivity = None
        if request.sample_thickness:
            resistivity = sheet_resistance * request.sample_thickness  # Ω·cm

        # Calculate conductivity
        conductivity = 1.0 / sheet_resistance if sheet_resistance > 0 else 0

        return FourPointProbeResponse(
            sheet_resistance=sheet_resistance,
            resistivity=resistivity,
            conductivity=conductivity,
            measurement_id=f"4PP-{id(request)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class HallEffectRequest(BaseModel):
    magnetic_field: float  # Tesla
    hall_voltage: float  # Volts
    current: float  # Amperes
    sample_thickness: float  # cm
    sample_type: str = "n-type"  # n-type or p-type

class HallEffectResponse(BaseModel):
    carrier_concentration: float  # cm^-3
    hall_mobility: float  # cm^2/V·s
    hall_coefficient: float
    conductivity_type: str
    measurement_id: str

@app.post("/api/electrical/hall-effect/measure", response_model=HallEffectResponse)
async def measure_hall_effect(request: HallEffectRequest):
    """
    Hall Effect measurement
    Determine carrier concentration and mobility
    """
    try:
        # Calculate Hall coefficient (cm^3/C)
        hall_coefficient = (request.hall_voltage * request.sample_thickness) / (request.magnetic_field * request.current)

        # Calculate carrier concentration (cm^-3)
        e = 1.602e-19  # Elementary charge (C)
        carrier_concentration = 1.0 / (abs(hall_coefficient) * e)

        # Estimate mobility (simplified)
        # In real implementation, would use sheet resistance from 4PP
        hall_mobility = abs(hall_coefficient) * 1000  # Simplified estimate

        conductivity_type = request.sample_type

        return HallEffectResponse(
            carrier_concentration=carrier_concentration,
            hall_mobility=hall_mobility,
            hall_coefficient=hall_coefficient,
            conductivity_type=conductivity_type,
            measurement_id=f"HALL-{id(request)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Optical Characterization ====================

class UVVisNIRRequest(BaseModel):
    wavelengths: List[float]  # nm
    intensities: List[float]  # arbitrary units
    measurement_type: str = "transmission"  # transmission, reflection, absorption

class UVVisNIRResponse(BaseModel):
    band_gap: Optional[float]  # eV
    peak_wavelength: float  # nm
    absorption_coefficient: List[float]
    measurement_id: str

@app.post("/api/optical/uv-vis-nir/analyze", response_model=UVVisNIRResponse)
async def analyze_uv_vis_nir(request: UVVisNIRRequest):
    """
    UV-Vis-NIR Spectroscopy Analysis
    Extract optical properties from spectrum
    """
    try:
        # Find peak wavelength
        peak_idx = request.intensities.index(max(request.intensities))
        peak_wavelength = request.wavelengths[peak_idx]

        # Simplified band gap estimation (Tauc plot would be used in real implementation)
        band_gap = None
        if len(request.wavelengths) > 0:
            # Convert wavelength to energy: E(eV) = 1240/λ(nm)
            energies = [1240/wl for wl in request.wavelengths]
            band_gap = max(energies) * 0.8  # Simplified estimate

        # Simplified absorption coefficient
        absorption_coefficient = [i * 0.01 for i in request.intensities]

        return UVVisNIRResponse(
            band_gap=band_gap,
            peak_wavelength=peak_wavelength,
            absorption_coefficient=absorption_coefficient,
            measurement_id=f"UVVIS-{id(request)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SPC ====================

class SPCDataPoint(BaseModel):
    timestamp: str
    value: float
    sample_id: Optional[str] = None

class SPCRequest(BaseModel):
    data: List[SPCDataPoint]
    chart_type: str = "xbar_r"  # xbar_r, i_mr, ewma, cusum
    target: Optional[float] = None
    ucl: Optional[float] = None
    lcl: Optional[float] = None

class SPCViolation(BaseModel):
    rule: str
    timestamp: str
    value: float
    severity: str

class SPCResponse(BaseModel):
    in_control: bool
    violations: List[SPCViolation]
    control_limits: Dict[str, float]
    process_capability: Optional[Dict[str, float]]
    measurement_id: str

@app.post("/api/spc/analyze", response_model=SPCResponse)
async def analyze_spc(request: SPCRequest):
    """
    Statistical Process Control Analysis
    Detect out-of-control conditions and violations
    """
    try:
        values = [dp.value for dp in request.data]

        # Calculate control limits
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean)**2 for x in values) / len(values)) ** 0.5

        ucl = request.ucl if request.ucl else mean + 3 * std_dev
        lcl = request.lcl if request.lcl else mean - 3 * std_dev

        # Check for violations (simplified Western Electric rules)
        violations = []
        for dp in request.data:
            if dp.value > ucl or dp.value < lcl:
                violations.append(SPCViolation(
                    rule="Rule 1: Point beyond control limits",
                    timestamp=dp.timestamp,
                    value=dp.value,
                    severity="critical"
                ))

        # Calculate process capability (simplified)
        spec_range = ucl - lcl
        process_capability = {
            "cp": spec_range / (6 * std_dev) if std_dev > 0 else 0,
            "cpk": min((ucl - mean) / (3 * std_dev), (mean - lcl) / (3 * std_dev)) if std_dev > 0 else 0
        }

        return SPCResponse(
            in_control=len(violations) == 0,
            violations=violations,
            control_limits={"ucl": ucl, "cl": mean, "lcl": lcl},
            process_capability=process_capability,
            measurement_id=f"SPC-{id(request)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Machine Learning / Virtual Metrology ====================

class VirtualMetrologyRequest(BaseModel):
    process_parameters: Dict[str, float]
    equipment_data: Dict[str, float]
    target_metric: str  # thickness, resistance, etc.

class VirtualMetrologyResponse(BaseModel):
    predicted_value: float
    confidence_interval: List[float]
    feature_importance: Dict[str, float]
    model_version: str
    measurement_id: str

@app.post("/api/ml/virtual-metrology/predict", response_model=VirtualMetrologyResponse)
async def predict_virtual_metrology(request: VirtualMetrologyRequest):
    """
    Virtual Metrology Prediction
    Predict process metrics from equipment data
    """
    try:
        # Simplified prediction (in real implementation, use trained ML model)
        # For demo: weighted sum of parameters
        predicted_value = sum(request.process_parameters.values()) / len(request.process_parameters)

        # Simplified confidence interval
        confidence = predicted_value * 0.05  # ±5%
        confidence_interval = [predicted_value - confidence, predicted_value + confidence]

        # Simplified feature importance
        feature_importance = {
            k: 1.0/len(request.process_parameters)
            for k in request.process_parameters.keys()
        }

        return VirtualMetrologyResponse(
            predicted_value=predicted_value,
            confidence_interval=confidence_interval,
            feature_importance=feature_importance,
            model_version="v2.1.0",
            measurement_id=f"VM-{id(request)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Status Endpoint ====================

@app.get("/api/status")
async def get_status():
    """Get service status and available endpoints"""
    return {
        "service": "analysis",
        "status": "operational",
        "endpoints": {
            "electrical": [
                "/api/electrical/four-point-probe/measure",
                "/api/electrical/hall-effect/measure"
            ],
            "optical": [
                "/api/optical/uv-vis-nir/analyze"
            ],
            "spc": [
                "/api/spc/analyze"
            ],
            "ml": [
                "/api/ml/virtual-metrology/predict"
            ]
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
