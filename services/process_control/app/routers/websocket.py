"""WebSocket endpoints for real-time telemetry streaming.

Provides WebSocket connections for:
- Ion Implantation: Beam current, pressure, field, dose, 2D profiles, alerts
- RTP: Setpoint vs measured temperature, lamp power, ramp error, alerts

Clients connect to /api/ion/stream/{run_id} or /api/rtp/stream/{run_id}
and receive JSON messages with telemetry updates.
"""

import asyncio
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from fastapi.exceptions import WebSocketException

from app.models.job import JobStore, JobType, JobStatus as JobStatusEnum
from app.auth import decode_token, require_org_access, User, Permission


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter()


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self):
        self.active_connections: Dict[str, list] = {}

    async def connect(self, run_id: str, websocket: WebSocket):
        """Accept WebSocket connection and add to active connections."""
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

    def disconnect(self, run_id: str, websocket: WebSocket):
        """Remove WebSocket connection."""
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def send_message(self, run_id: str, message: dict):
        """Send message to all connected clients for a run."""
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error sending message: {e}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for run_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error broadcasting message: {e}")


manager = ConnectionManager()


# ============================================================================
# Authentication for WebSocket
# ============================================================================

async def authenticate_websocket(websocket: WebSocket, token: Optional[str]) -> Optional[User]:
    """Authenticate WebSocket connection using JWT token.

    Args:
        websocket: WebSocket connection
        token: JWT token from query parameter

    Returns:
        User object if authenticated, None otherwise

    Raises:
        WebSocketException: If authentication fails
    """
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    try:
        token_data = decode_token(token)
        from app.auth import Role, ROLE_PERMISSIONS

        role = Role(token_data.role)
        permissions = ROLE_PERMISSIONS.get(role, [])

        return User(
            user_id=token_data.user_id,
            email=token_data.email,
            org_id=token_data.org_id,
            role=role,
            permissions=permissions,
        )
    except Exception as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)


# ============================================================================
# Telemetry Data Loading
# ============================================================================

async def load_telemetry_from_storage(run_id: str, job_type: JobType) -> Optional[Dict[str, Any]]:
    """Load telemetry data from storage (filesystem or MinIO).

    Args:
        run_id: Run identifier
        job_type: ION_IMPLANT or RTP

    Returns:
        Telemetry dictionary or None if not found
    """
    telemetry_storage = os.getenv("TELEMETRY_STORAGE", "/tmp/spectra/telemetry")

    # Find telemetry file for run_id
    prefix = "ion" if job_type == JobType.ION_IMPLANT else "rtp"

    try:
        files = [f for f in os.listdir(telemetry_storage) if f.startswith(f"{prefix}_{run_id}")]
        if not files:
            return None

        # Load most recent file
        latest_file = sorted(files)[-1]
        filepath = os.path.join(telemetry_storage, latest_file)

        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading telemetry: {e}")
        return None


# ============================================================================
# Ion Implantation WebSocket
# ============================================================================

@router.websocket("/api/ion/stream/{run_id}")
async def stream_ion_telemetry(
    websocket: WebSocket,
    run_id: str,
    token: Optional[str] = Query(None),
):
    """Stream real-time telemetry for Ion Implantation run.

    **Connection**: `ws://localhost:8003/api/ion/stream/{run_id}?token=JWT_TOKEN`

    **Authentication**: JWT token required in query parameter

    **Required permission**: `ion:view_telemetry`

    **Message format**:
    ```json
    {
        "type": "progress" | "telemetry" | "alert" | "completed" | "error",
        "timestamp": "2024-01-15T10:30:00Z",
        "data": {...}
    }
    ```

    **Message types**:

    1. **progress**: Job progress update
       ```json
       {
           "type": "progress",
           "data": {
               "progress": 45.2,
               "current_step": "Implanting (45% dose)",
               "status": "running"
           }
       }
       ```

    2. **telemetry**: Real-time measurements
       ```json
       {
           "type": "telemetry",
           "data": {
               "time_s": 10.5,
               "beam_current_ma": 5.02,
               "chamber_pressure_torr": 2.1e-6,
               "analyzer_field_v": 40000,
               "integrated_dose_atoms_cm2": 4.5e14,
               "dose_uniformity_pct": 95.2,
               "wafer_temp_c": 55.3
           }
       }
       ```

    3. **alert**: SPC alert
       ```json
       {
           "type": "alert",
           "data": {
               "parameter": "beam_current_ma",
               "message": "Beam current beyond 3σ control limits",
               "severity": "critical"
           }
       }
       ```

    4. **completed**: Run finished
       ```json
       {
           "type": "completed",
           "data": {
               "final_dose_atoms_cm2": 1.02e15,
               "dose_error_pct": 2.1,
               "vm_prediction": {...}
           }
       }
       ```

    5. **error**: Run failed
       ```json
       {
           "type": "error",
           "data": {
               "error_message": "Beam stability check failed"
           }
       }
       ```
    """

    # Authenticate
    user = await authenticate_websocket(websocket, token)

    # Check permission
    if Permission.ION_VIEW_TELEMETRY not in user.permissions:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Find job
    job = JobStore.get_job_by_run_id(run_id)
    if not job:
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    # Check org access
    if job.org_id != user.org_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Connect
    await manager.connect(run_id, websocket)

    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "run_id": run_id,
                "status": job.status.value,
                "progress": job.progress,
            },
        })

        # Stream telemetry
        while True:
            # Get current job status
            job = JobStore.get_job_by_run_id(run_id)

            if not job:
                await websocket.send_json({
                    "type": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"error_message": "Job not found"},
                })
                break

            # Send progress update
            await websocket.send_json({
                "type": "progress",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "progress": job.progress,
                    "current_step": job.current_step,
                    "status": job.status.value,
                },
            })

            # If job is completed or failed, send final message and close
            if job.status in [JobStatusEnum.COMPLETED, JobStatusEnum.FAILED, JobStatusEnum.CANCELLED]:
                if job.status == JobStatusEnum.COMPLETED:
                    # Load telemetry from storage
                    telemetry = await load_telemetry_from_storage(run_id, JobType.ION_IMPLANT)

                    await websocket.send_json({
                        "type": "completed",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "final_dose_atoms_cm2": job.metadata.get("final_dose_atoms_cm2"),
                            "dose_error_pct": job.metadata.get("dose_error_pct"),
                            "vm_prediction": job.metadata.get("vm_prediction"),
                            "artifacts": job.artifacts,
                        },
                    })

                    # Send telemetry data (sampled for efficiency)
                    if telemetry:
                        # Send every 10th data point
                        sampled_telemetry = {
                            key: value[::10] if isinstance(value, list) and len(value) > 10 else value
                            for key, value in telemetry.items()
                        }

                        await websocket.send_json({
                            "type": "telemetry",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": sampled_telemetry,
                        })

                elif job.status == JobStatusEnum.FAILED:
                    await websocket.send_json({
                        "type": "error",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"error_message": job.error_message},
                    })

                elif job.status == JobStatusEnum.CANCELLED:
                    await websocket.send_json({
                        "type": "cancelled",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"message": "Run was cancelled"},
                    })

                break

            # Wait before next update
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        manager.disconnect(run_id, websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(run_id, websocket)


# ============================================================================
# RTP WebSocket
# ============================================================================

@router.websocket("/api/rtp/stream/{run_id}")
async def stream_rtp_telemetry(
    websocket: WebSocket,
    run_id: str,
    token: Optional[str] = Query(None),
):
    """Stream real-time telemetry for RTP run.

    **Connection**: `ws://localhost:8003/api/rtp/stream/{run_id}?token=JWT_TOKEN`

    **Authentication**: JWT token required in query parameter

    **Required permission**: `rtp:view_telemetry`

    **Message format**:
    ```json
    {
        "type": "progress" | "telemetry" | "alert" | "completed" | "error",
        "timestamp": "2024-01-15T10:30:00Z",
        "data": {...}
    }
    ```

    **Message types**:

    1. **progress**: Job progress update
       ```json
       {
           "type": "progress",
           "data": {
               "progress": 60.0,
               "current_step": "RTP segment 2/4 (985.3°C)",
               "status": "running"
           }
       }
       ```

    2. **telemetry**: Real-time measurements
       ```json
       {
           "type": "telemetry",
           "data": {
               "time_s": 25.0,
               "setpoint_temp_c": 1000.0,
               "measured_temp_c": 998.5,
               "lamp_power_pct": 75.3,
               "ramp_error_c": 1.5,
               "chamber_pressure_torr": 1.0,
               "n2_flow_slm": 10.0,
               "o2_flow_slm": 0.0
           }
       }
       ```

    3. **alert**: SPC alert
       ```json
       {
           "type": "alert",
           "data": {
               "parameter": "ramp_error_c",
               "message": "Ramp error exceeds 5°C",
               "severity": "warning"
           }
       }
       ```

    4. **completed**: Run finished
       ```json
       {
           "type": "completed",
           "data": {
               "final_temp_c": 1000.2,
               "temp_error_c": 0.2,
               "max_overshoot_c": 3.5,
               "thermal_budget": 5.2e5,
               "vm_prediction": {...}
           }
       }
       ```
    """

    # Authenticate
    user = await authenticate_websocket(websocket, token)

    # Check permission
    if Permission.RTP_VIEW_TELEMETRY not in user.permissions:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Find job
    job = JobStore.get_job_by_run_id(run_id)
    if not job:
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return

    # Check org access
    if job.org_id != user.org_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Connect
    await manager.connect(run_id, websocket)

    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "run_id": run_id,
                "status": job.status.value,
                "progress": job.progress,
            },
        })

        # Stream telemetry
        while True:
            # Get current job status
            job = JobStore.get_job_by_run_id(run_id)

            if not job:
                await websocket.send_json({
                    "type": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"error_message": "Job not found"},
                })
                break

            # Send progress update
            await websocket.send_json({
                "type": "progress",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "progress": job.progress,
                    "current_step": job.current_step,
                    "status": job.status.value,
                },
            })

            # If job is completed or failed, send final message
            if job.status in [JobStatusEnum.COMPLETED, JobStatusEnum.FAILED, JobStatusEnum.CANCELLED]:
                if job.status == JobStatusEnum.COMPLETED:
                    # Load telemetry from storage
                    telemetry = await load_telemetry_from_storage(run_id, JobType.RTP)

                    await websocket.send_json({
                        "type": "completed",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "final_temp_c": job.metadata.get("final_temp_c"),
                            "temp_error_c": job.metadata.get("temp_error_c"),
                            "max_overshoot_c": job.metadata.get("max_overshoot_c"),
                            "thermal_budget": job.metadata.get("thermal_budget"),
                            "vm_prediction": job.metadata.get("vm_prediction"),
                            "artifacts": job.artifacts,
                        },
                    })

                    # Send telemetry data (sampled)
                    if telemetry:
                        sampled_telemetry = {
                            key: value[::10] if isinstance(value, list) and len(value) > 10 else value
                            for key, value in telemetry.items()
                        }

                        await websocket.send_json({
                            "type": "telemetry",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": sampled_telemetry,
                        })

                elif job.status == JobStatusEnum.FAILED:
                    await websocket.send_json({
                        "type": "error",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"error_message": job.error_message},
                    })

                elif job.status == JobStatusEnum.CANCELLED:
                    await websocket.send_json({
                        "type": "cancelled",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"message": "Run was cancelled"},
                    })

                break

            # Wait before next update
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        manager.disconnect(run_id, websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(run_id, websocket)


# Export
__all__ = ["router", "manager"]
