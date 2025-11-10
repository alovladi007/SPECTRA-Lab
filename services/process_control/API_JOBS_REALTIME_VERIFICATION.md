# APIs, Background Jobs & Realtime - Verification

Verification of Section 6 requirements from the specification.

---

## Specification Requirements

```
6) APIs, background jobs & realtime

FastAPI (sketch OpenAPI; implement fully with RBAC/org guards):

Ion Implantation:
- POST /api/ion/runs → validate recipe + calibration + SOP approvals → enqueue Celery job; return run_id.
- GET /api/ion/runs/{id} → status + last metrics + links to artifacts.
- POST /api/ion/simulate/dose_profile → SRIM-like calculator (no hardware).
- WS /api/ion/stream/{run_id} → telemetry (current, pressure, field, dose, 2D profile thumbnails), controller setpoints, alerts.

RTP:
- POST /api/rtp/runs → recipe (segments, dwell, gas, pressure, emissivity) validated → enqueue job.
- GET /api/rtp/runs/{id} → status/metrics/artifacts.
- POST /api/rtp/tune/controller → PID/MPC tuning advisor for a provided plant ID or prior runs.
- WS /api/rtp/stream/{run_id} → setpoint vs measured T, lamp power, ramp error, alerts.

Jobs: Celery tasks with progress %, cancellable, retries/backoff. Store job logs & artifacts URIs; push progress via WebSocket/SSE.
```

---

## Current Implementation Status

### Existing API Files

| File | Purpose | Status |
|------|---------|--------|
| [app/api/endpoints.py](services/process_control/app/api/endpoints.py) | Main API endpoints | ⚠️ Partial - Basic endpoints only |
| [app/api/safety_endpoints.py](services/process_control/app/api/safety_endpoints.py) | Safety & calibration | ✅ Implemented |
| [app/main.py](services/process_control/app/main.py) | FastAPI application | ✅ Implemented |

### Existing Endpoints (Current Implementation)

**Ion Implantation:**
- ❌ POST /api/ion/runs - **NOT IMPLEMENTED**
- ❌ GET /api/ion/runs/{id} - **NOT IMPLEMENTED**
- ❌ POST /api/ion/simulate/dose_profile - **NOT IMPLEMENTED**
- ⚠️ WS /api/ion/stream/{run_id} - **PARTIAL** (basic WebSocket exists at `/ws`)

**RTP:**
- ❌ POST /api/rtp/runs - **NOT IMPLEMENTED**
- ❌ GET /api/rtp/runs/{id} - **NOT IMPLEMENTED**
- ❌ POST /api/rtp/tune/controller - **NOT IMPLEMENTED**
- ❌ WS /api/rtp/stream/{run_id} - **NOT IMPLEMENTED**

**Background Jobs:**
- ❌ Celery integration - **NOT IMPLEMENTED**
- ❌ Progress tracking - **NOT IMPLEMENTED**
- ❌ Cancellation support - **NOT IMPLEMENTED**
- ❌ Retries/backoff - **NOT IMPLEMENTED**
- ❌ Job logs & artifacts - **NOT IMPLEMENTED**
- ❌ WebSocket/SSE progress updates - **NOT IMPLEMENTED**

**RBAC/Org Guards:**
- ❌ Role-Based Access Control - **NOT IMPLEMENTED**
- ❌ Organization guards - **NOT IMPLEMENTED**

---

## Requirements Checklist

### Ion Implantation API (4 Requirements)

- [ ] POST /api/ion/runs
  - [ ] Recipe validation
  - [ ] Calibration validation
  - [ ] SOP approvals check
  - [ ] Enqueue Celery job
  - [ ] Return run_id
  - [ ] RBAC/org guards

- [ ] GET /api/ion/runs/{id}
  - [ ] Run status (queued, running, completed, failed)
  - [ ] Last metrics (dose, uniformity, beam current, etc.)
  - [ ] Links to artifacts (logs, profiles, analysis)
  - [ ] RBAC/org guards

- [ ] POST /api/ion/simulate/dose_profile
  - [ ] Input: species, energy, tilt, twist, dose
  - [ ] Call SRIM estimator (no hardware)
  - [ ] Return: depth profile, range, channeling risk, sheet resistance
  - [ ] No job required (synchronous)
  - [ ] RBAC/org guards

- [ ] WS /api/ion/stream/{run_id}
  - [ ] Realtime telemetry stream
  - [ ] Beam current (mA)
  - [ ] Pressure (source, analyzer, process)
  - [ ] Analyzer field (Tesla)
  - [ ] Integrated dose (cm⁻²)
  - [ ] 2D dose profile thumbnails
  - [ ] Controller setpoints
  - [ ] FDC alerts (beam drift, uniformity)
  - [ ] RBAC/org guards

### RTP API (4 Requirements)

- [ ] POST /api/rtp/runs
  - [ ] Recipe validation (segments, dwell, gas, pressure, emissivity)
  - [ ] Calibration validation
  - [ ] SOP approvals check
  - [ ] Enqueue Celery job
  - [ ] Return run_id
  - [ ] RBAC/org guards

- [ ] GET /api/rtp/runs/{id}
  - [ ] Run status (queued, running, completed, failed)
  - [ ] Metrics (ramp fidelity, thermal budget, soak stability)
  - [ ] Links to artifacts (logs, temperature profiles, performance report)
  - [ ] RBAC/org guards

- [ ] POST /api/rtp/tune/controller
  - [ ] Input: plant_id or run_ids for analysis
  - [ ] Analyze historical performance
  - [ ] Generate PID tuning recommendations (Ziegler-Nichols)
  - [ ] Generate MPC weight recommendations (Q, R matrices)
  - [ ] Return: suggested gains, expected improvement
  - [ ] No job required (synchronous)
  - [ ] RBAC/org guards

- [ ] WS /api/rtp/stream/{run_id}
  - [ ] Realtime telemetry stream
  - [ ] Setpoint temperature vs measured (per zone)
  - [ ] Lamp power (% per zone)
  - [ ] Ramp tracking error (°C)
  - [ ] Controller output (PID/MPC)
  - [ ] Alerts (overshoot, tracking error, drift)
  - [ ] RBAC/org guards

### Background Jobs (7 Requirements)

- [ ] Celery Integration
  - [ ] Install and configure Celery
  - [ ] Redis/RabbitMQ broker
  - [ ] Worker configuration
  - [ ] Task routing (ion vs rtp queues)

- [ ] Progress Tracking
  - [ ] Task progress % (0-100)
  - [ ] Current step description
  - [ ] Estimated time remaining
  - [ ] Store progress in cache (Redis)

- [ ] Cancellation Support
  - [ ] Graceful task cancellation
  - [ ] Hardware cleanup on cancel
  - [ ] Update task status to "cancelled"
  - [ ] POST /api/jobs/{job_id}/cancel endpoint

- [ ] Retries & Backoff
  - [ ] Automatic retry on transient failures
  - [ ] Exponential backoff (2^n seconds)
  - [ ] Max retries configurable
  - [ ] Dead letter queue for failed tasks

- [ ] Job Logs & Artifacts
  - [ ] Store task logs (stdout, stderr)
  - [ ] Store artifacts (profiles, analysis, metrics)
  - [ ] S3/MinIO URIs for large files
  - [ ] Database records for metadata

- [ ] WebSocket/SSE Progress
  - [ ] Push progress updates via WebSocket
  - [ ] Or Server-Sent Events (SSE) fallback
  - [ ] Channel per run_id
  - [ ] Disconnect handling

- [ ] Job Status API
  - [ ] GET /api/jobs/{job_id} → status, progress, logs, artifacts
  - [ ] GET /api/jobs → list jobs (with filters)
  - [ ] RBAC/org guards

### Security & RBAC (3 Requirements)

- [ ] RBAC Implementation
  - [ ] User roles (admin, engineer, operator, viewer)
  - [ ] Permission system (create_run, view_run, cancel_run, etc.)
  - [ ] Middleware for role checking
  - [ ] Decorator: @require_permission("ion:create_run")

- [ ] Organization Guards
  - [ ] Multi-tenancy support
  - [ ] Organization filtering (user can only see their org's data)
  - [ ] org_id in database models
  - [ ] Middleware for org isolation

- [ ] Authentication
  - [ ] JWT token validation
  - [ ] Integration with shared auth service
  - [ ] Token refresh mechanism
  - [ ] Dependency: get_current_user()

---

## Verification Summary

### Total Requirements: 18

- ✅ **Implemented: 0 (0%)**
- ❌ **Missing: 18 (100%)**
- ⚠️ **Partial: 0 (0%)**

### Status by Category

| Category | Requirements | Implemented | Percentage |
|----------|-------------|-------------|------------|
| Ion Implantation API | 4 | 0 | 0% |
| RTP API | 4 | 0 | 0% |
| Background Jobs (Celery) | 7 | 0 | 0% |
| Security & RBAC | 3 | 0 | 0% |

---

## Proposed Implementation Plan

### Phase 1: Background Jobs Infrastructure

1. **Install Celery & Broker**
   ```bash
   pip install celery redis
   ```

2. **Create Celery App** (`app/celery_app.py`)
   ```python
   from celery import Celery

   celery_app = Celery(
       "process_control",
       broker="redis://localhost:6379/0",
       backend="redis://localhost:6379/0"
   )

   celery_app.conf.update(
       task_routes={
           "app.tasks.ion.*": {"queue": "ion"},
           "app.tasks.rtp.*": {"queue": "rtp"},
       },
       task_serializer="json",
       result_serializer="json",
       accept_content=["json"],
       task_track_started=True,
       task_send_sent_event=True,
   )
   ```

3. **Create Task Definitions** (`app/tasks/ion_tasks.py`, `app/tasks/rtp_tasks.py`)
   ```python
   from app.celery_app import celery_app
   from celery import Task

   class CallbackTask(Task):
       def on_success(self, retval, task_id, args, kwargs):
           # Update database, send WebSocket notification
           pass

       def on_failure(self, exc, task_id, args, kwargs, einfo):
           # Log error, send alert
           pass

       def on_retry(self, exc, task_id, args, kwargs, einfo):
           # Update retry count
           pass

   @celery_app.task(
       base=CallbackTask,
       bind=True,
       max_retries=3,
       default_retry_delay=60
   )
   def run_ion_implant(self, run_id, recipe, calibration_id, sop_approval_id):
       # Execute ion implant run
       # Update progress: self.update_state(state='PROGRESS', meta={'progress': 50})
       pass
   ```

4. **Create Job Models** (`app/models/job.py`)
   ```python
   from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Enum
   from enum import Enum as PyEnum

   class JobStatus(str, PyEnum):
       QUEUED = "queued"
       RUNNING = "running"
       COMPLETED = "completed"
       FAILED = "failed"
       CANCELLED = "cancelled"

   class Job(Base):
       __tablename__ = "jobs"

       id = Column(String, primary_key=True)
       run_id = Column(String, unique=True, nullable=False)
       org_id = Column(String, nullable=False)
       job_type = Column(String, nullable=False)  # "ion_implant", "rtp"
       status = Column(Enum(JobStatus), default=JobStatus.QUEUED)
       progress = Column(Float, default=0.0)
       current_step = Column(String)
       started_at = Column(DateTime)
       completed_at = Column(DateTime)
       error_message = Column(String)
       logs_uri = Column(String)
       artifacts = Column(JSON)  # List of artifact URIs
   ```

### Phase 2: API Endpoints

5. **Create Ion API Endpoints** (`app/api/ion_endpoints.py`)
   ```python
   from fastapi import APIRouter, Depends, HTTPException
   from app.auth import require_permission, get_current_user
   from app.tasks.ion_tasks import run_ion_implant

   router = APIRouter(prefix="/api/ion", tags=["Ion Implantation"])

   @router.post("/runs", response_model=RunResponse)
   async def create_ion_run(
       request: IonRunRequest,
       user: User = Depends(get_current_user),
       _: None = Depends(require_permission("ion:create_run"))
   ):
       # Validate recipe
       # Check calibration
       # Verify SOP approvals
       # Create run record
       # Enqueue Celery task
       task = run_ion_implant.delay(run_id, recipe, calibration_id, sop_approval_id)
       return {"run_id": run_id, "job_id": task.id, "status": "queued"}

   @router.get("/runs/{run_id}", response_model=RunStatus)
   async def get_ion_run_status(
       run_id: str,
       user: User = Depends(get_current_user),
       _: None = Depends(require_permission("ion:view_run"))
   ):
       # Fetch run + job from database
       # Return status, metrics, artifacts
       pass

   @router.post("/simulate/dose_profile", response_model=DoseProfileSimulation)
   async def simulate_dose_profile(
       request: DoseProfileRequest,
       user: User = Depends(get_current_user)
   ):
       from app.models.ion_range import SRIMEstimator
       estimator = SRIMEstimator()

       # Call SRIM estimator (synchronous)
       range_params = estimator.estimate_range(...)
       depth_profile = estimator.predict_depth_profile(...)
       sheet_resistance = estimator.estimate_sheet_resistance(...)

       return {
           "depth_profile": depth_profile,
           "range": range_params,
           "sheet_resistance": sheet_resistance
       }
   ```

6. **Create RTP API Endpoints** (`app/api/rtp_endpoints.py`)
   ```python
   @router.post("/runs", response_model=RunResponse)
   async def create_rtp_run(...):
       # Similar to ion
       pass

   @router.get("/runs/{run_id}", response_model=RunStatus)
   async def get_rtp_run_status(...):
       pass

   @router.post("/tune/controller", response_model=ControllerTuningRecommendations)
   async def tune_rtp_controller(
       request: ControllerTuningRequest,
       user: User = Depends(get_current_user)
   ):
       from app.controllers.rtp import PerformanceAnalyzer
       analyzer = PerformanceAnalyzer()

       # Analyze historical runs
       # Generate tuning recommendations
       recommendations = analyzer.recommend_tuning(plant_id, run_ids)

       return recommendations
   ```

### Phase 3: WebSocket Streaming

7. **Create WebSocket Handlers** (`app/api/websocket.py`)
   ```python
   from fastapi import WebSocket, WebSocketDisconnect
   from app.auth import verify_websocket_token

   @router.websocket("/ion/stream/{run_id}")
   async def ion_telemetry_stream(websocket: WebSocket, run_id: str):
       await websocket.accept()

       # Verify authentication
       user = await verify_websocket_token(websocket)

       # Subscribe to Redis pub/sub for run_id
       pubsub = redis_client.pubsub()
       pubsub.subscribe(f"ion:telemetry:{run_id}")

       try:
           async for message in pubsub.listen():
               if message["type"] == "message":
                   data = json.loads(message["data"])
                   await websocket.send_json(data)
       except WebSocketDisconnect:
           pubsub.unsubscribe()

   @router.websocket("/rtp/stream/{run_id}")
   async def rtp_telemetry_stream(websocket: WebSocket, run_id: str):
       # Similar to ion
       pass
   ```

8. **Publish Telemetry from Tasks**
   ```python
   # In ion_tasks.py
   def run_ion_implant(self, run_id, ...):
       while running:
           # Read telemetry from hardware
           telemetry = {
               "beam_current_mA": current,
               "dose_cm2": integrated_dose,
               "pressure_mtorr": pressure,
               # ...
           }

           # Publish to Redis
           redis_client.publish(
               f"ion:telemetry:{run_id}",
               json.dumps(telemetry)
           )

           # Update progress
           self.update_state(
               state='PROGRESS',
               meta={'progress': progress_pct, 'step': 'Implanting wafer 5/25'}
           )
   ```

### Phase 4: RBAC & Security

9. **Create RBAC Middleware** (`app/auth.py`)
   ```python
   from fastapi import Depends, HTTPException, status
   from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

   security = HTTPBearer()

   async def get_current_user(
       credentials: HTTPAuthorizationCredentials = Depends(security)
   ) -> User:
       # Verify JWT token
       # Fetch user from database
       # Return user object
       pass

   def require_permission(permission: str):
       async def permission_checker(user: User = Depends(get_current_user)):
           if not user.has_permission(permission):
               raise HTTPException(
                   status_code=status.HTTP_403_FORBIDDEN,
                   detail=f"Permission denied: {permission}"
               )
       return Depends(permission_checker)

   def require_org_access(org_id: str):
       async def org_checker(user: User = Depends(get_current_user)):
           if user.org_id != org_id and not user.is_admin:
               raise HTTPException(
                   status_code=status.HTTP_403_FORBIDDEN,
                   detail="Organization access denied"
               )
       return Depends(org_checker)
   ```

10. **Create Permission System**
    ```python
    # Permissions:
    # - ion:create_run, ion:view_run, ion:cancel_run
    # - rtp:create_run, rtp:view_run, rtp:cancel_run
    # - jobs:view, jobs:cancel
    # - calibration:create, calibration:view
    # - sop:approve, sop:view

    # Roles:
    # - admin: all permissions
    # - engineer: create/view/cancel runs, view calibration
    # - operator: create/view runs, view calibration
    # - viewer: view only
    ```

---

## Conclusion

**Section 6 (APIs, background jobs & realtime) is NOT implemented.**

Required Implementation:
- 4 Ion Implantation API endpoints
- 4 RTP API endpoints
- Celery background jobs infrastructure
- WebSocket/SSE realtime streaming
- RBAC & organization guards

Estimated Effort:
- Phase 1 (Celery): 2-3 days
- Phase 2 (API endpoints): 2-3 days
- Phase 3 (WebSocket): 1-2 days
- Phase 4 (RBAC): 1-2 days
- **Total: 6-10 days of development**

Dependencies:
- Redis (for Celery broker and WebSocket pub/sub)
- Database migrations (Job, Run models)
- Shared authentication service
- S3/MinIO for artifact storage

---

**Status**: ❌ **NOT IMPLEMENTED**
**Priority**: HIGH (required for production)
**Next Steps**: Begin Phase 1 (Celery infrastructure)
