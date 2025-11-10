"""FastAPI endpoints for RTP and Ion Implantation modules."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, WebSocket
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import asyncio
import json

from app.core.deps import get_db, get_current_user, require_role
from app.models.modules import (
    ImplantDoseProfile, ImplantTelemetry,
    RTPProfile, RTPTelemetry,
    SPCSeries, SPCPoint, SPCAlert,
    VMFeatureSet, VMModel
)
from app.schemas.modules import (
    ImplantDoseProfileCreate, ImplantDoseProfileResponse,
    ImplantTelemetryCreate, ImplantTelemetryResponse,
    RTPProfileCreate, RTPProfileResponse,
    RTPTelemetryCreate, RTPTelemetryResponse,
    SPCSeriesCreate, SPCSeriesResponse,
    SPCPointCreate, SPCPointResponse,
    SPCAlertResponse, SPCAlertAcknowledge,
    VMFeatureSetCreate, VMFeatureSetResponse,
    VMModelCreate, VMModelResponse, VMModelDeploy,
    VMPrediction, VMPredictionResponse,
    TelemetryQuery, SPCAnalysis
)
from app.drivers.hardware import create_driver, DriverConfig
from app.simulators.hil import IonImplantSimulator, RTPSimulator, SimulatorConfig, SimulationMode
from app.control.algorithms import PIDController, MPCController, R2RController, AdaptiveController
from app.services.spc_vm import SPCService, VirtualMetrologyEngine

# Initialize services
spc_service = SPCService()
vm_engine = VirtualMetrologyEngine()

# Hardware drivers (initialized on startup)
drivers = {}
simulators = {}

# WebSocket connections for real-time updates
active_connections = []


# Router for Ion Implantation
implant_router = APIRouter(prefix="/api/v1/implant", tags=["Ion Implantation"])

@implant_router.post("/profiles", response_model=ImplantDoseProfileResponse)
async def create_implant_profile(
    profile: ImplantDoseProfileCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create new ion implantation dose profile."""
    db_profile = ImplantDoseProfile(
        org_id=current_user["org_id"],
        **profile.dict()
    )
    
    # Calculate SRIM parameters
    if 'implant' in simulators:
        simulator = simulators['implant']
        simulator.set_beam_parameters(
            profile.energy_keV,
            1.0,  # Default current
            profile.ion_species.value
        )
        Rp, dRp = simulator.physics.calculate_projected_range()
        db_profile.projected_range_nm = Rp
        db_profile.straggle_nm = dRp
    
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    
    # Start background uniformity calculation
    background_tasks.add_task(calculate_beam_uniformity, db_profile.id)
    
    return db_profile

@implant_router.get("/profiles/{profile_id}", response_model=ImplantDoseProfileResponse)
async def get_implant_profile(
    profile_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get ion implantation profile by ID."""
    profile = db.query(ImplantDoseProfile).filter(
        ImplantDoseProfile.id == profile_id,
        ImplantDoseProfile.org_id == current_user["org_id"]
    ).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return profile

@implant_router.post("/telemetry", response_model=ImplantTelemetryResponse)
async def record_implant_telemetry(
    telemetry: ImplantTelemetryCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Record ion implantation telemetry data."""
    db_telemetry = ImplantTelemetry(
        ts=datetime.now(),
        **telemetry.dict()
    )
    
    db.add(db_telemetry)
    db.commit()
    db.refresh(db_telemetry)
    
    # Send to SPC monitoring
    await spc_service.add_point(
        f"implant_current_{telemetry.run_id}",
        telemetry.beam_current_mA
    )
    
    # Broadcast to WebSocket clients
    await broadcast_telemetry("implant", db_telemetry.dict())
    
    return db_telemetry

@implant_router.get("/telemetry", response_model=List[ImplantTelemetryResponse])
async def query_implant_telemetry(
    query: TelemetryQuery = Depends(),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Query ion implantation telemetry data."""
    q = db.query(ImplantTelemetry)
    
    if query.run_id:
        q = q.filter(ImplantTelemetry.run_id == query.run_id)
    if query.start_time:
        q = q.filter(ImplantTelemetry.ts >= query.start_time)
    if query.end_time:
        q = q.filter(ImplantTelemetry.ts <= query.end_time)
    
    return q.order_by(ImplantTelemetry.ts.desc()).limit(query.limit).offset(query.offset).all()

@implant_router.post("/control/start")
async def start_implant(
    run_id: int,
    target_dose_cm2: float,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_role(["operator", "engineer", "admin"]))
):
    """Start ion implantation process."""
    if 'implant' in drivers:
        driver = drivers['implant']
        success = await driver.start_implant(target_dose_cm2)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start implant")
    
    if 'implant' in simulators:
        simulator = simulators['implant']
        simulator.start_implant(target_dose_cm2)
    
    return {"status": "started", "run_id": run_id, "target_dose": target_dose_cm2}

@implant_router.post("/control/stop")
async def stop_implant(
    run_id: int,
    current_user: dict = Depends(require_role(["operator", "engineer", "admin"]))
):
    """Stop ion implantation process."""
    if 'implant' in drivers:
        driver = drivers['implant']
        success = await driver.stop_implant()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to stop implant")
    
    if 'implant' in simulators:
        simulator = simulators['implant']
        simulator.stop_implant()
    
    return {"status": "stopped", "run_id": run_id}


# Router for RTP
rtp_router = APIRouter(prefix="/api/v1/rtp", tags=["Rapid Thermal Processing"])

@rtp_router.post("/profiles", response_model=RTPProfileResponse)
async def create_rtp_profile(
    profile: RTPProfileCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create new RTP temperature profile."""
    # Calculate peak temperature from recipe
    peak_T = max(segment.T_C for segment in profile.recipe_curve)
    
    db_profile = RTPProfile(
        org_id=current_user["org_id"],
        recipe_curve=[s.dict() for s in profile.recipe_curve],
        peak_T_C=peak_T,
        **{k: v for k, v in profile.dict().items() if k != 'recipe_curve'}
    )
    
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    
    return db_profile

@rtp_router.get("/profiles/{profile_id}", response_model=RTPProfileResponse)
async def get_rtp_profile(
    profile_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get RTP profile by ID."""
    profile = db.query(RTPProfile).filter(
        RTPProfile.id == profile_id,
        RTPProfile.org_id == current_user["org_id"]
    ).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return profile

@rtp_router.post("/telemetry", response_model=RTPTelemetryResponse)
async def record_rtp_telemetry(
    telemetry: RTPTelemetryCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Record RTP telemetry data."""
    db_telemetry = RTPTelemetry(
        ts=datetime.now(),
        **telemetry.dict()
    )
    
    db.add(db_telemetry)
    db.commit()
    db.refresh(db_telemetry)
    
    # Send to SPC monitoring
    await spc_service.add_point(
        f"rtp_temperature_{telemetry.run_id}",
        telemetry.pyrometer_T_C
    )
    
    # Broadcast to WebSocket clients
    await broadcast_telemetry("rtp", db_telemetry.dict())
    
    return db_telemetry

@rtp_router.get("/telemetry", response_model=List[RTPTelemetryResponse])
async def query_rtp_telemetry(
    query: TelemetryQuery = Depends(),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Query RTP telemetry data."""
    q = db.query(RTPTelemetry)
    
    if query.run_id:
        q = q.filter(RTPTelemetry.run_id == query.run_id)
    if query.start_time:
        q = q.filter(RTPTelemetry.ts >= query.start_time)
    if query.end_time:
        q = q.filter(RTPTelemetry.ts <= query.end_time)
    
    return q.order_by(RTPTelemetry.ts.desc()).limit(query.limit).offset(query.offset).all()

@rtp_router.post("/control/start-recipe")
async def start_rtp_recipe(
    profile_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_role(["operator", "engineer", "admin"]))
):
    """Start RTP recipe execution."""
    # Get profile
    profile = db.query(RTPProfile).filter(RTPProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    if 'rtp' in drivers:
        driver = drivers['rtp']
        await driver.set_recipe(profile.recipe_curve)
        success = await driver.start_recipe()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start recipe")
    
    if 'rtp' in simulators:
        simulator = simulators['rtp']
        simulator.set_recipe({'name': f'Profile_{profile_id}', 'segments': profile.recipe_curve})
        simulator.start_recipe()
    
    return {"status": "started", "profile_id": profile_id, "run_id": run_id}

@rtp_router.post("/control/stop-recipe")
async def stop_rtp_recipe(
    run_id: int,
    current_user: dict = Depends(require_role(["operator", "engineer", "admin"]))
):
    """Stop RTP recipe execution."""
    if 'rtp' in drivers:
        driver = drivers['rtp']
        success = await driver.stop_recipe()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to stop recipe")
    
    if 'rtp' in simulators:
        simulator = simulators['rtp']
        simulator.stop_recipe()
    
    return {"status": "stopped", "run_id": run_id}


# Router for SPC
spc_router = APIRouter(prefix="/api/v1/spc", tags=["Statistical Process Control"])

@spc_router.post("/series", response_model=SPCSeriesResponse)
async def create_spc_series(
    series: SPCSeriesCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create new SPC monitoring series."""
    db_series = SPCSeries(
        org_id=current_user["org_id"],
        control_limits=series.control_limits.dict(),
        rules=[r.dict() for r in series.rules],
        **{k: v for k, v in series.dict().items() if k not in ['control_limits', 'rules']}
    )
    
    db.add(db_series)
    db.commit()
    db.refresh(db_series)
    
    return db_series

@spc_router.get("/series", response_model=List[SPCSeriesResponse])
async def list_spc_series(
    active_only: bool = Query(True),
    instrument_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """List SPC series."""
    q = db.query(SPCSeries).filter(SPCSeries.org_id == current_user["org_id"])
    
    if active_only:
        q = q.filter(SPCSeries.active == True)
    if instrument_id:
        q = q.filter(SPCSeries.instrument_id == instrument_id)
    
    return q.all()

@spc_router.post("/points", response_model=SPCPointResponse)
async def add_spc_point(
    point: SPCPointCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Add new SPC data point."""
    # Verify series exists and belongs to user's org
    series = db.query(SPCSeries).filter(
        SPCSeries.id == point.series_id,
        SPCSeries.org_id == current_user["org_id"]
    ).first()
    
    if not series:
        raise HTTPException(status_code=404, detail="SPC series not found")
    
    # Process through SPC service
    result = await spc_service.add_point(
        str(point.series_id),
        point.value,
        datetime.now()
    )
    
    # Store in database
    db_point = SPCPoint(
        ts=datetime.now(),
        violations=result.get('violations'),
        **point.dict()
    )
    
    db.add(db_point)
    db.commit()
    db.refresh(db_point)
    
    # Create alerts if violations detected
    if result.get('violations'):
        background_tasks.add_task(
            create_spc_alerts,
            db_point.id,
            point.series_id,
            result['violations'],
            current_user["org_id"]
        )
    
    return db_point

@spc_router.get("/points/{series_id}", response_model=List[SPCPointResponse])
async def get_spc_points(
    series_id: int,
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get SPC points for a series."""
    # Verify series belongs to user's org
    series = db.query(SPCSeries).filter(
        SPCSeries.id == series_id,
        SPCSeries.org_id == current_user["org_id"]
    ).first()
    
    if not series:
        raise HTTPException(status_code=404, detail="SPC series not found")
    
    return db.query(SPCPoint).filter(
        SPCPoint.series_id == series_id
    ).order_by(SPCPoint.ts.desc()).limit(limit).all()

@spc_router.get("/alerts", response_model=List[SPCAlertResponse])
async def get_spc_alerts(
    acknowledged: Optional[bool] = Query(None),
    series_id: Optional[int] = Query(None),
    limit: int = Query(50),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get SPC alerts."""
    q = db.query(SPCAlert).filter(SPCAlert.org_id == current_user["org_id"])
    
    if acknowledged is not None:
        q = q.filter(SPCAlert.acknowledged == acknowledged)
    if series_id:
        q = q.filter(SPCAlert.series_id == series_id)
    
    return q.order_by(SPCAlert.created_at.desc()).limit(limit).all()

@spc_router.post("/alerts/{alert_id}/acknowledge", response_model=SPCAlertResponse)
async def acknowledge_spc_alert(
    alert_id: int,
    ack: SPCAlertAcknowledge,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge SPC alert."""
    alert = db.query(SPCAlert).filter(
        SPCAlert.id == alert_id,
        SPCAlert.org_id == current_user["org_id"]
    ).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged = True
    alert.acknowledged_by = current_user["user_id"]
    alert.acknowledged_at = datetime.now()
    alert.resolution_notes = ack.resolution_notes
    
    db.commit()
    db.refresh(alert)
    
    return alert

@spc_router.post("/analysis", response_model=Dict[str, Any])
async def run_spc_analysis(
    analysis: SPCAnalysis,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Run comprehensive SPC analysis."""
    # Verify series belongs to user's org
    series = db.query(SPCSeries).filter(
        SPCSeries.id == analysis.series_id,
        SPCSeries.org_id == current_user["org_id"]
    ).first()
    
    if not series:
        raise HTTPException(status_code=404, detail="SPC series not found")
    
    # Get statistics from service
    stats = spc_service.get_series_statistics(str(analysis.series_id))
    
    # Add database statistics
    q = db.query(SPCPoint).filter(SPCPoint.series_id == analysis.series_id)
    
    if analysis.start_time:
        q = q.filter(SPCPoint.ts >= analysis.start_time)
    if analysis.end_time:
        q = q.filter(SPCPoint.ts <= analysis.end_time)
    
    points = q.all()
    values = [p.value for p in points]
    
    if values:
        stats.update({
            'points_analyzed': len(values),
            'violations_found': sum(1 for p in points if p.violations),
            'time_range': {
                'start': min(p.ts for p in points).isoformat(),
                'end': max(p.ts for p in points).isoformat()
            }
        })
    
    return stats


# Router for Virtual Metrology
vm_router = APIRouter(prefix="/api/v1/vm", tags=["Virtual Metrology"])

@vm_router.post("/feature-sets", response_model=VMFeatureSetResponse)
async def create_feature_set(
    feature_set: VMFeatureSetCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create VM feature set."""
    db_feature_set = VMFeatureSet(
        org_id=current_user["org_id"],
        features=[f.dict() for f in feature_set.features],
        **{k: v for k, v in feature_set.dict().items() if k != 'features'}
    )
    
    db.add(db_feature_set)
    db.commit()
    db.refresh(db_feature_set)
    
    return db_feature_set

@vm_router.post("/models", response_model=VMModelResponse)
async def create_vm_model(
    model: VMModelCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create VM model."""
    # Verify feature set exists and belongs to user's org
    feature_set = db.query(VMFeatureSet).filter(
        VMFeatureSet.id == model.feature_set_id,
        VMFeatureSet.org_id == current_user["org_id"]
    ).first()
    
    if not feature_set:
        raise HTTPException(status_code=404, detail="Feature set not found")
    
    db_model = VMModel(
        org_id=current_user["org_id"],
        **model.dict()
    )
    
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    return db_model

@vm_router.post("/models/{model_id}/deploy", response_model=VMModelResponse)
async def deploy_vm_model(
    model_id: int,
    deploy: VMModelDeploy,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_role(["engineer", "admin"]))
):
    """Deploy/undeploy VM model."""
    model = db.query(VMModel).filter(
        VMModel.id == model_id,
        VMModel.org_id == current_user["org_id"]
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model.deployed = deploy.deployed
    if deploy.deployed:
        model.approved_by = current_user["user_id"]
        model.approved_at = datetime.now()
    
    db.commit()
    db.refresh(model)
    
    # Register/unregister with VM engine
    if deploy.deployed and model.model_uri:
        vm_engine.register_model(
            f"model_{model_id}",
            model.model_uri
        )
    
    return model

@vm_router.post("/predict", response_model=VMPredictionResponse)
async def predict_vm(
    prediction: VMPrediction,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Make VM prediction."""
    # Verify model exists and is deployed
    model = db.query(VMModel).filter(
        VMModel.id == prediction.model_id,
        VMModel.org_id == current_user["org_id"],
        VMModel.deployed == True
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found or not deployed")
    
    # Get telemetry data for run
    telemetry = db.query(RTPTelemetry).filter(
        RTPTelemetry.run_id == prediction.run_id
    ).order_by(RTPTelemetry.ts.desc()).limit(100).all()
    
    if not telemetry:
        raise HTTPException(status_code=404, detail="No telemetry data found for run")
    
    # Convert to DataFrame for prediction
    import pandas as pd
    telemetry_df = pd.DataFrame([t.dict() for t in telemetry])
    
    # Make prediction
    try:
        result = vm_engine.predict(f"model_{model.id}", telemetry_df)
        
        return VMPredictionResponse(
            run_id=prediction.run_id,
            model_id=prediction.model_id,
            predictions=result['prediction'],
            confidence_intervals=result.get('confidence_interval'),
            feature_importance=result.get('feature_importance'),
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# WebSocket endpoint for real-time telemetry
@rtp_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry streaming."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)


# Helper functions
async def broadcast_telemetry(module: str, data: dict):
    """Broadcast telemetry to all connected WebSocket clients."""
    message = json.dumps({
        "module": module,
        "type": "telemetry",
        "data": data,
        "timestamp": datetime.now().isoformat()
    })
    
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            # Remove dead connections
            active_connections.remove(connection)

async def calculate_beam_uniformity(profile_id: int):
    """Background task to calculate beam uniformity."""
    # Simulate calculation
    await asyncio.sleep(2)
    
    # In production, would update database with results
    print(f"Calculated uniformity for profile {profile_id}")

async def create_spc_alerts(point_id: int, series_id: int, violations: List[dict], org_id: int):
    """Create SPC alerts for violations."""
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    try:
        for violation in violations:
            alert = SPCAlert(
                org_id=org_id,
                series_id=series_id,
                point_id=point_id,
                alert_type=violation['rule'],
                severity="warning" if "1_sigma" in violation['rule'] else "critical",
                rule_violated=violation['rule'],
                description=violation['description']
            )
            db.add(alert)
        
        db.commit()
    finally:
        db.close()


# Startup event to initialize hardware
async def startup_hardware():
    """Initialize hardware drivers and simulators on startup."""
    # Initialize simulators
    sim_config = SimulatorConfig(mode=SimulationMode.REALISTIC)
    
    implant_sim = IonImplantSimulator(sim_config)
    await implant_sim.start()
    simulators['implant'] = implant_sim
    
    rtp_sim = RTPSimulator(sim_config)
    await rtp_sim.start()
    simulators['rtp'] = rtp_sim
    
    # Initialize hardware drivers (if configured)
    # In production, would read from configuration
    if False:  # Set to True when hardware is available
        implant_driver = create_driver(
            'ion_implant',
            DriverConfig(
                name="Ion Implanter",
                connection_string="TCPIP::192.168.1.100::5025::SOCKET"
            )
        )
        await implant_driver.connect()
        drivers['implant'] = implant_driver
        
        rtp_driver = create_driver(
            'rtp',
            DriverConfig(
                name="RTP System",
                connection_string="opc.tcp://192.168.1.101:4840"
            )
        )
        await rtp_driver.connect()
        drivers['rtp'] = rtp_driver
    
    print("Hardware initialization complete")


# Shutdown event to cleanup hardware
async def shutdown_hardware():
    """Cleanup hardware connections on shutdown."""
    # Stop simulators
    for simulator in simulators.values():
        await simulator.stop()
    
    # Disconnect drivers
    for driver in drivers.values():
        await driver.disconnect()
    
    print("Hardware cleanup complete")
