# Implementation Guide: Complete Diffusion Module Integration
**Status:** Partial - Database schema and env files created
**Date:** November 9, 2025

---

## ✅ COMPLETED

1. **Database Schema** (`deployment/init-db.sql`)
   - ✅ simulation_audit table
   - ✅ batch_jobs table
   - ✅ kpi_measurements table
   - ✅ spc_violations table
   - ✅ maintenance_recommendations table
   - ✅ calibration_results table
   - ✅ Views and indexes

2. **Environment Configuration**
   - ✅ `.env.example` created
   - ✅ `.env.development` created

3. **Frontend Pages** (all created previously)
   - ✅ All 6 simulation pages
   - ✅ Navigation integrated

4. **Recharts Library**
   - ✅ Already installed in package.json

---

## 🔧 REMAINING TASKS

### 1. Start Docker Services

```bash
# Navigate to deployment directory
cd Diffusion_Module_Complete/session12/deployment

# Start Postgres, Redis, MinIO
docker compose up -d postgres redis minio

# Verify services are running
docker compose ps

# Check logs
docker compose logs postgres
```

**Expected Output:**
```
NAME                  STATUS          PORTS
diffusion-postgres    Up 30 seconds   0.0.0.0:5432->5432/tcp
diffusion-redis       Up 30 seconds   0.0.0.0:6379->6379/tcp
diffusion-minio       Up 30 seconds   0.0.0.0:9000-9001->9000-9001/tcp
```

---

###  2. Add SQLAlchemy Models

**File:** `services/analysis/app/simulation/models.py`

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class SimulationAudit(Base):
    __tablename__ = 'simulation_audit'

    id = Column(Integer, primary_key=True)
    simulation_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    run_id = Column(String(100))
    recipe_id = Column(String(100))
    simulation_type = Column(String(50), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    user_id = Column(String(100))
    user_email = Column(String(255))

    parameters = Column(JSONB, nullable=False)
    results = Column(JSONB)

    status = Column(String(20), default='pending')
    execution_time_ms = Column(Integer)
    memory_mb = Column(Float)

    error_message = Column(Text)
    error_traceback = Column(Text)

    metadata = Column(JSONB)
    tags = Column(ARRAY(Text))

    git_commit = Column(String(40))
    module_version = Column(String(20))

class BatchJob(Base):
    __tablename__ = 'batch_jobs'

    id = Column(Integer, primary_key=True)
    job_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    job_name = Column(String(255))

    simulation_type = Column(String(50), nullable=False)
    total_simulations = Column(Integer, nullable=False)
    completed_simulations = Column(Integer, default=0)
    failed_simulations = Column(Integer, default=0)

    status = Column(String(20), default='queued')

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    user_id = Column(String(100))
    config = Column(JSONB)
    results_summary = Column(JSONB)
    error_message = Column(Text)

class KPIMeasurement(Base):
    __tablename__ = 'kpi_measurements'

    id = Column(Integer, primary_key=True)
    measurement_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    simulation_id = Column(UUID(as_uuid=True))

    measured_at = Column(DateTime(timezone=True), server_default=func.now())

    kpi_name = Column(String(100), nullable=False)
    kpi_value = Column(Float, nullable=False)
    kpi_unit = Column(String(50))

    target_value = Column(Float)
    ucl = Column(Float)
    lcl = Column(Float)
    usl = Column(Float)
    lsl = Column(Float)

    within_control = Column(Boolean)
    within_spec = Column(Boolean)

    tool_id = Column(String(100))
    recipe_id = Column(String(100))
    wafer_id = Column(String(100))
    lot_id = Column(String(100))

    metadata = Column(JSONB)
```

---

### 3. Add Database Connection

**File:** `services/analysis/app/simulation/database.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import os
from contextlib import contextmanager

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/diffusion"
)

engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # For development
    echo=True if os.getenv("ENVIRONMENT") == "development" else False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db() -> Session:
    """Database session context manager"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def save_simulation(simulation_data: dict) -> str:
    """Save simulation to database"""
    from .models import SimulationAudit

    with get_db() as db:
        sim = SimulationAudit(**simulation_data)
        db.add(sim)
        db.flush()
        return str(sim.simulation_id)

def get_simulation(simulation_id: str):
    """Retrieve simulation from database"""
    from .models import SimulationAudit

    with get_db() as db:
        return db.query(SimulationAudit).filter(
            SimulationAudit.simulation_id == simulation_id
        ).first()
```

---

### 4. Update API to Use Database

**File:** `services/analysis/app/api/v1/simulation/routers.py`

**Find this code:**
```python
# In-memory storage (will be replaced with database in production)
jobs_db: Dict[str, Dict] = {}
results_db: Dict[str, Dict] = {}
```

**Replace with:**
```python
from app.simulation.database import save_simulation, get_simulation

# Remove in-memory dicts, use database instead
```

**Update the diffusion endpoint:**
```python
@router.post("/diffusion", response_model=DiffusionResponse)
async def run_diffusion_simulation(request: DiffusionRequest):
    try:
        simulation_id = str(uuid.uuid4())
        start_time = time.time()

        # ... existing simulation code ...

        # Save to database instead of memory
        save_simulation({
            "simulation_id": simulation_id,
            "simulation_type": "diffusion",
            "parameters": request.dict(),
            "results": {
                "junction_depth": xj,
                "sheet_resistance": Rs,
                # ... other results
            },
            "status": "completed",
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "module_version": "1.12.0"
        })

        # Return response...
```

---

### 5. Add Chart Components (Recharts)

#### Diffusion Profile Chart Component

**File:** `apps/web/src/components/charts/DiffusionProfileChart.tsx`

```typescript
'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface DiffusionProfileChartProps {
  depth: number[]
  concentration: number[]
}

export function DiffusionProfileChart({ depth, concentration }: DiffusionProfileChartProps) {
  const data = depth.map((d, i) => ({
    depth: d,
    concentration: concentration[i]
  }))

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
        <XAxis
          dataKey="depth"
          label={{ value: 'Depth (nm)', position: 'insideBottom', offset: -5 }}
          className="text-gray-700 dark:text-gray-300"
        />
        <YAxis
          scale="log"
          domain={['auto', 'auto']}
          label={{ value: 'Concentration (atoms/cm³)', angle: -90, position: 'insideLeft' }}
          className="text-gray-700 dark:text-gray-300"
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'rgb(31 41 55)',
            border: '1px solid rgb(55 65 81)',
            borderRadius: '0.5rem'
          }}
          formatter={(value: number) => value.toExponential(2)}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="concentration"
          stroke="#8b5cf6"
          strokeWidth={2}
          dot={false}
          name="Dopant Concentration"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

#### SPC Control Chart Component

**File:** `apps/web/src/components/charts/SPCControlChart.tsx`

```typescript
'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'

interface SPCControlChartProps {
  dataPoints: number[]
  centerline: number
  ucl: number
  lcl: number
}

export function SPCControlChart({ dataPoints, centerline, ucl, lcl }: SPCControlChartProps) {
  const data = dataPoints.map((value, index) => ({
    sample: index + 1,
    value,
    centerline,
    ucl,
    lcl
  }))

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-gray-600" />
        <XAxis
          dataKey="sample"
          label={{ value: 'Sample Number', position: 'insideBottom', offset: -5 }}
        />
        <YAxis label={{ value: 'Measurement', angle: -90, position: 'insideLeft' }} />
        <Tooltip />
        <Legend />

        {/* Control limits */}
        <ReferenceLine y={ucl} stroke="#ef4444" strokeDasharray="3 3" label="UCL" />
        <ReferenceLine y={centerline} stroke="#22c55e" strokeDasharray="3 3" label="CL" />
        <ReferenceLine y={lcl} stroke="#ef4444" strokeDasharray="3 3" label="LCL" />

        {/* Data line */}
        <Line
          type="monotone"
          dataKey="value"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={{ fill: '#3b82f6', r: 4 }}
          name="Measurement"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

---

### 6. Update Frontend Pages to Use Charts

#### Update Diffusion Page

**File:** `apps/web/src/app/dashboard/simulation/diffusion/page.tsx`

**Import the chart:**
```typescript
import { DiffusionProfileChart } from '@/components/charts/DiffusionProfileChart'
```

**Replace the placeholder div (line ~243) with:**
```typescript
<DiffusionProfileChart
  depth={results.profile.depth}
  concentration={results.profile.concentration}
/>
```

#### Update SPC Page

**File:** `apps/web/src/app/dashboard/simulation/spc/page.tsx`

**Import:**
```typescript
import { SPCControlChart } from '@/components/charts/SPCControlChart'
```

**Replace placeholder (line ~244) with:**
```typescript
<SPCControlChart
  dataPoints={results.data_points}
  centerline={results.centerline}
  ucl={results.ucl}
  lcl={results.lcl}
/>
```

---

## 📦 Installation Steps

```bash
# 1. Start Docker services
cd Diffusion_Module_Complete/session12/deployment
docker compose up -d postgres redis minio

# 2. Install Python database dependencies
cd ../../..
pip install sqlalchemy psycopg2-binary

# 3. Create chart components directory
mkdir -p apps/web/src/components/charts

# 4. Restart backend API
# (It will auto-reload if watching for changes)

# 5. Test database connection
psql postgresql://postgres:postgres@localhost:5432/diffusion -c "SELECT * FROM simulation_audit LIMIT 1;"
```

---

## ✅ Verification

After implementation, verify:

1. **Database:**
   ```bash
   docker compose ps | grep postgres  # Should show "Up"
   psql postgresql://postgres:postgres@localhost:5432/diffusion -c "\dt"
   ```

2. **API with Database:**
   ```bash
   curl -X POST http://localhost:8001/api/v1/simulation/diffusion \
     -H "Content-Type: application/json" \
     -d '{"temperature": 1000, "time": 30, "dopant": "boron"}'

   # Check database
   psql postgresql://postgres:postgres@localhost:5432/diffusion \
     -c "SELECT simulation_id, simulation_type, status FROM simulation_audit;"
   ```

3. **Charts:**
   - Visit http://localhost:3012/dashboard/simulation/diffusion
   - Run a simulation
   - Verify chart displays instead of placeholder

---

## 🎯 Success Criteria

- [ ] Docker containers running (postgres, redis, minio)
- [ ] Database tables created from init-db.sql
- [ ] SQLAlchemy models defined
- [ ] API saves to database instead of memory
- [ ] Recharts components created
- [ ] All placeholder charts replaced with actual visualizations
- [ ] Frontend displays real-time charts

---

## 📞 Troubleshooting

**Issue: Docker containers won't start**
```bash
docker compose logs postgres
# Check for port conflicts on 5432
lsof -i :5432
```

**Issue: Database connection fails**
```bash
# Check DATABASE_URL in .env
# Verify postgres is accessible
psql postgresql://postgres:postgres@localhost:5432/diffusion
```

**Issue: Charts don't render**
```bash
# Check browser console for errors
# Verify recharts is installed
npm list recharts
```

---

**Estimated Time to Complete:** 2-3 hours
**Complexity:** Moderate
**Status:** Ready for implementation
