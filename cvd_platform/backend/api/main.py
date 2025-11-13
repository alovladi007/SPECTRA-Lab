"""
Main FastAPI Application for CVD Platform
Provides RESTful API and WebSocket endpoints for:
- Data acquisition and monitoring
- Process control and recipe management
- Virtual metrology and predictions
- SPC/FDC monitoring
- Analytics and reporting
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import logging
import asyncio

# Import CVD platform modules
import sys
sys.path.append('..')

from data_acquisition.sensor_interface import SensorFactory, SensorReading, SensorType
from physics_models.cvd_reactor_model import CVDReactorModel, ReactorGeometry, ReactorDimensions, ProcessConditions
from virtual_metrology.vm_predictor import VirtualMetrologyPredictor, DesignFeatures, ProcessFeatures
from process_control.r2r_controller import APCController, RecipeParameters, ControlTarget
from spc_fdc.spc_monitor import SPCManager, ChartType
from analytics.anomaly_detector import AnomalyDetectionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CVD Platform API",
    description="Advanced Chemical Vapor Deposition Control and Analytics Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (in production, use dependency injection and proper state management)
spc_manager = SPCManager()
apc_controller = APCController()
vm_predictor = VirtualMetrologyPredictor(model_type="lightgbm")
anomaly_engine = AnomalyDetectionEngine()

# Initialize CVD reactor model
reactor_dimensions = ReactorDimensions(
    length=1.0,
    diameter=0.5,
    susceptor_diameter=0.35,
    wafer_diameter=0.3,
    gap_height=0.05,
    heater_zones=5,
    inlet_diameter=0.05,
    outlet_diameter=0.05
)
reactor_model = CVDReactorModel(ReactorGeometry.SHOWERHEAD, reactor_dimensions)

# Pydantic models for API


class ProcessConditionsRequest(BaseModel):
    temperature: float
    pressure: float
    gas_flows: Dict[str, float]
    rotation_speed: float
    deposition_time: float
    susceptor_temp: float
    wall_temp: float


class VMPredictionRequest(BaseModel):
    design_features: Dict[str, Any]
    process_features: Dict[str, Any]


class RecipeUpdateRequest(BaseModel):
    current_recipe: Dict[str, Any]
    target_thickness: float
    measured_thickness: float
    wafer_number: int


class SPCMeasurementRequest(BaseModel):
    chart_id: str
    value: float
    timestamp: Optional[str] = None


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CVD Platform API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "up",
            "spc": "up",
            "apc": "up",
            "vm": "up",
            "analytics": "up"
        }
    }


# Sensor & Data Acquisition Endpoints

@app.get("/sensors/list")
async def list_sensors():
    """List available sensor types"""
    sensor_types = [s.value for s in SensorType]
    return {"sensor_types": sensor_types}


@app.post("/sensors/reading")
async def get_sensor_reading(sensor_type: str, sensor_id: str):
    """Get current sensor reading"""
    try:
        sensor = SensorFactory.create_sensor(sensor_type, sensor_id, {})
        await sensor.connect()
        reading = await sensor.read()
        await sensor.disconnect()

        return reading.to_dict()
    except Exception as e:
        logger.error(f"Error reading sensor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Physics Model & Digital Twin Endpoints

@app.post("/simulation/run")
async def run_simulation(conditions: ProcessConditionsRequest):
    """Run CVD reactor simulation"""
    try:
        process_conditions = ProcessConditions(
            temperature=conditions.temperature,
            pressure=conditions.pressure,
            gas_flows=conditions.gas_flows,
            rotation_speed=conditions.rotation_speed,
            deposition_time=conditions.deposition_time,
            susceptor_temp=conditions.susceptor_temp,
            wall_temp=conditions.wall_temp
        )

        result = reactor_model.run_full_simulation(process_conditions)

        return {
            "status": "success",
            "thickness": result["thickness"],
            "deposition_rate": result["deposition_rate"].tolist() if result["deposition_rate"] is not None else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/simulation/status")
async def get_simulation_status():
    """Get digital twin status"""
    return {
        "reactor_type": reactor_model.geometry.value,
        "dimensions": reactor_dimensions.__dict__,
        "mesh_size": f"{reactor_model.nr} x {reactor_model.nz}",
        "status": "ready"
    }


# Virtual Metrology Endpoints

@app.post("/vm/predict")
async def predict_thickness(request: VMPredictionRequest):
    """Predict film thickness using virtual metrology"""
    try:
        # Convert request to feature objects
        design_feat = DesignFeatures(**request.design_features)
        process_feat = ProcessFeatures(**request.process_features)

        # Predict
        prediction = vm_predictor.predict_thickness(design_feat, process_feat)

        return prediction.to_dict()

    except Exception as e:
        logger.error(f"VM prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vm/update")
async def update_vm_model(wafer_id: str, actual_thickness: float):
    """Update VM model with actual metrology"""
    try:
        vm_predictor.update_with_metrology(wafer_id, actual_thickness)
        return {
            "status": "success",
            "message": f"VM model updated with {wafer_id} measurement"
        }
    except Exception as e:
        logger.error(f"VM update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Process Control Endpoints

@app.post("/control/recipe-update")
async def update_recipe(request: RecipeUpdateRequest):
    """Calculate APC recipe update"""
    try:
        # Parse current recipe
        current_recipe = RecipeParameters(**request.current_recipe)

        # Define target
        target = ControlTarget(
            target_thickness=request.target_thickness,
            thickness_tolerance=5.0,
            target_uniformity=2.0,
            uniformity_tolerance=0.5
        )

        # Calculate update
        updated_recipe = apc_controller.calculate_recipe_update(
            current_recipe,
            target,
            request.measured_thickness,
            vm_prediction=None,
            wafer_number=request.wafer_number
        )

        return {
            "status": "success",
            "updated_recipe": updated_recipe.__dict__,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Recipe update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/control/status")
async def get_control_status():
    """Get APC controller status"""
    return {
        "controllers": {
            "r2r": "active",
            "mpc": "active",
            "adaptive": "active",
            "drift_compensator": "active"
        },
        "status": "operational"
    }


# SPC/FDC Endpoints

@app.post("/spc/chart/create")
async def create_spc_chart(
    chart_id: str,
    parameter_name: str,
    chart_type: str = "xbar",
    target: Optional[float] = None,
    lsl: Optional[float] = None,
    usl: Optional[float] = None
):
    """Create new SPC control chart"""
    try:
        chart_type_enum = ChartType(chart_type)
        spec_limits = (lsl, usl) if lsl and usl else None

        chart = spc_manager.create_chart(
            chart_id=chart_id,
            parameter_name=parameter_name,
            chart_type=chart_type_enum,
            target=target,
            spec_limits=spec_limits
        )

        return {
            "status": "success",
            "chart_id": chart_id,
            "chart_type": chart_type
        }

    except Exception as e:
        logger.error(f"Chart creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/spc/measurement/add")
async def add_spc_measurement(request: SPCMeasurementRequest):
    """Add measurement to SPC chart"""
    try:
        timestamp = datetime.fromisoformat(request.timestamp) if request.timestamp else None

        violations = spc_manager.add_measurement(
            chart_id=request.chart_id,
            value=request.value,
            timestamp=timestamp
        )

        return {
            "status": "success",
            "violations": [v.__dict__ for v in violations]
        }

    except Exception as e:
        logger.error(f"Measurement add error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/spc/charts/summary")
async def get_spc_summary():
    """Get summary of all SPC charts"""
    try:
        summary = spc_manager.get_chart_summary()
        return summary
    except Exception as e:
        logger.error(f"SPC summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/spc/alarms/active")
async def get_active_alarms(lookback_hours: int = 24):
    """Get active SPC alarms"""
    try:
        alarms = spc_manager.get_active_alarms(lookback_hours)
        return {
            "count": len(alarms),
            "alarms": [alarm.__dict__ for alarm in alarms]
        }
    except Exception as e:
        logger.error(f"Alarm retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics & Anomaly Detection Endpoints

@app.post("/analytics/anomaly/detect")
async def detect_anomalies(data: Dict[str, float]):
    """Detect anomalies in process data"""
    try:
        # Convert dict to numpy array
        import numpy as np
        feature_names = list(data.keys())
        data_array = np.array([list(data.values())])

        # Detect
        anomalies = anomaly_engine.detect_anomalies(
            data=data_array[0],
            feature_names=feature_names,
            timestamp=datetime.utcnow()
        )

        return {
            "status": "success",
            "anomalies_detected": len(anomalies),
            "anomalies": [a.__dict__ for a in anomalies]
        }

    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/maintenance/predictions")
async def get_maintenance_predictions(equipment_id: str):
    """Get predictive maintenance recommendations"""
    try:
        # Example prediction
        prediction = anomaly_engine.predictive_maintenance.predict_failure(
            equipment_id=equipment_id,
            component="heater",
            current_health=0.75,
            degradation_rate=0.0001
        )

        return prediction.__dict__

    except Exception as e:
        logger.error(f"Maintenance prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time data streaming

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time data streaming"""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Send simulated real-time data
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "temperature": 800.0 + np.random.randn() * 5.0,
                "pressure": 10.0 + np.random.randn() * 0.5,
                "deposition_rate": 5.0 + np.random.randn() * 0.2
            }

            await websocket.send_json(data)
            await asyncio.sleep(1.0)  # 1 Hz update rate

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket client disconnected")


# Recipe Management Endpoints

@app.get("/recipes/list")
async def list_recipes():
    """List available recipes"""
    # In production, retrieve from database
    recipes = [
        {"recipe_id": "CVD_Si_001", "name": "Silicon Deposition 100nm", "temperature": 800, "time": 120},
        {"recipe_id": "CVD_Si_002", "name": "Silicon Deposition 200nm", "temperature": 820, "time": 240},
        {"recipe_id": "CVD_SiN_001", "name": "Silicon Nitride 50nm", "temperature": 750, "time": 180},
    ]
    return {"recipes": recipes}


@app.get("/recipes/{recipe_id}")
async def get_recipe(recipe_id: str):
    """Get recipe details"""
    # In production, retrieve from database
    recipe = {
        "recipe_id": recipe_id,
        "name": "Silicon Deposition 100nm",
        "parameters": {
            "temperature": 800,
            "pressure": 10,
            "precursor_flow": 100,
            "carrier_flow": 5000,
            "deposition_time": 120,
            "rotation_speed": 20
        },
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-15T10:30:00"
    }
    return recipe


@app.post("/recipes/create")
async def create_recipe(recipe_data: Dict[str, Any]):
    """Create new recipe"""
    # In production, save to database
    return {
        "status": "success",
        "recipe_id": "CVD_NEW_001",
        "message": "Recipe created successfully"
    }


# Equipment Status Endpoints

@app.get("/equipment/status")
async def get_equipment_status():
    """Get overall equipment status"""
    return {
        "equipment_id": "CVD-01",
        "status": "processing",
        "current_wafer": 15,
        "lot_id": "LOT-2024-001",
        "recipe": "CVD_Si_001",
        "uptime_hours": 156.5,
        "last_pm": "2024-01-01T00:00:00",
        "next_pm": "2024-02-01T00:00:00"
    }


# Reports & Analytics Endpoints

@app.get("/reports/daily-summary")
async def get_daily_summary(date: str):
    """Get daily production summary"""
    return {
        "date": date,
        "wafers_processed": 125,
        "average_thickness": 98.5,
        "uniformity": 1.8,
        "yield": 98.2,
        "violations": 3,
        "chamber_uptime": 22.5
    }


if __name__ == "__main__":
    import uvicorn
    import numpy as np

    # Initialize some example SPC charts
    spc_manager.create_chart(
        chart_id="thickness_chart",
        parameter_name="Film Thickness",
        chart_type=ChartType.XBAR,
        target=100.0,
        spec_limits=(95.0, 105.0)
    )

    logger.info("Starting CVD Platform API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
