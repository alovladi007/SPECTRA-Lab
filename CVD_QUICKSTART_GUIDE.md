# CVD Platform - Quick Start Guide

**Status:** ✅ Backend Integration Fixed | ✅ Sample Data Available | ✅ Ready to Use

---

## What Was Fixed

### Issues Resolved ✅

1. **❌ CVD Router Import Error**
   - **Before:** `from ..database import get_db` (incorrect path)
   - **After:** `from services.shared.db.deps import get_db` ✅

2. **❌ Double API Prefix**
   - **Before:** Router had `/api/v1/cvd` + main.py added `/api/v1` = `/api/v1/api/v1/cvd` ❌
   - **After:** Router registered without extra prefix = `/api/v1/cvd` ✅

3. **❌ CVD Models Not Imported**
   - **Before:** CVD tables not created on startup
   - **After:** `from app.models import cvd` added to main.py ✅

4. **❌ No Sample Data**
   - **Before:** Empty database, frontend showed zeros
   - **After:** Comprehensive seed script with realistic data ✅

---

## Quick Start (3 Steps)

### Step 1: Start the Platform

```bash
# Start all services
docker-compose up -d

# Check services are running
docker-compose ps

# Expected services:
# - spectra-analysis (port 8001)
# - spectra-celery-worker
# - spectra-web (port 3012)
# - spectra-db (port 5433)
# - spectra-redis (port 6379)
```

### Step 2: Seed Sample Data

```bash
# Run the seeding script inside the container
docker-compose exec analysis python scripts/seed_cvd_data.py

# Expected output:
# ============================================================
# CVD Platform - Database Seeding
# ============================================================
# Creating CVD tables...
# ✓ Tables created
#
# Seeding Process Modes...
# ✓ Created 4 process modes
#
# Seeding Recipes...
# ✓ Created 3 recipes
#
# Seeding Runs...
# ✓ Created 9 runs
#
# Seeding SPC Series...
# ✓ Created 2 SPC series
# ============================================================
```

### Step 3: Access the Platform

```bash
# CVD Workspace
open http://localhost:3012/cvd/workspace

# API Documentation
open http://localhost:8001/docs

# CVD API Endpoints
open http://localhost:8001/api/v1/cvd
```

---

## What You'll See After Seeding

### Frontend (http://localhost:3012/cvd/workspace)

**Overview Tab:**
- **Active Runs:** 1 (currently processing)
- **Process Modes:** 4 (LPCVD, PECVD, MOCVD, AACVD)
- **Recipes:** 3 (across all modes)
- **Completed Today:** Variable based on run dates

**Recent Runs Table:**
| Run ID | Status | Lot ID | Wafers | Duration | Created |
|--------|--------|--------|--------|----------|---------|
| PECVD-2024099 | PROCESSING | LOT-CURRENT | 25 | - | Just now |
| PECVD-2024012 | COMPLETED | LOT-PECVD-3 | 50 | 910s | 2 days ago |
| PECVD-2024011 | COMPLETED | LOT-PECVD-2 | 50 | 910s | 3 days ago |
| ... | ... | ... | ... | ... | ... |

---

## Sample Data Details

### Process Modes (4 total)

#### 1. LPCVD Si3N4
- **Pressure Mode:** Low Pressure CVD
- **Energy:** Thermal
- **Temperature:** 650-850°C
- **Pressure:** 20-200 Pa
- **Materials:** Si3N4, SiO2, Poly-Si
- **Capacity:** 150 wafers

#### 2. PECVD SiO2
- **Pressure Mode:** Plasma-Enhanced CVD
- **Energy:** Plasma (13.56 MHz RF)
- **Temperature:** 250-400°C
- **Pressure:** 100-1000 Pa
- **Materials:** SiO2, SiON, SiN
- **Plasma Power:** 100-1000 W

#### 3. MOCVD GaN
- **Pressure Mode:** Low Pressure CVD
- **Energy:** Thermal
- **Temperature:** 900-1150°C
- **Pressure:** 10,000-40,000 Pa (100-400 Torr)
- **Materials:** GaN, InGaN, AlGaN
- **Rotation:** 100-1500 RPM

#### 4. AACVD ZnO
- **Pressure Mode:** Atmospheric Pressure CVD
- **Energy:** Thermal
- **Temperature:** 350-550°C
- **Pressure:** 101,325 Pa (1 atm)
- **Materials:** ZnO, SnO2, TiO2
- **Aerosol:** Ultrasonic (1.7 MHz)

---

### Recipes (3 total)

#### 1. LPCVD Si3N4 Standard
- **Process Mode:** LPCVD Si3N4
- **Target Thickness:** 100 nm
- **Uniformity:** 95%
- **Process Time:** 35 minutes
- **Temperature:** 780°C (3 zones)
- **Gases:** SiH4 (80 sccm), NH3 (160 sccm), N2 carrier (2000 sccm)
- **Pressure:** 133 Pa
- **Tags:** si3n4, passivation, lpcvd
- **Status:** ✅ Baseline Recipe

#### 2. PECVD SiO2 ILD
- **Process Mode:** PECVD SiO2
- **Target Thickness:** 500 nm
- **Uniformity:** 92%
- **Process Time:** 15 minutes
- **Temperature:** 350°C
- **Gases:** SiH4 (150 sccm), N2O (1000 sccm), N2 carrier (500 sccm)
- **Pressure:** 400 Pa
- **Plasma:** 300 W, 13.56 MHz, 100% duty cycle
- **Tags:** sio2, ild, pecvd, oxide
- **Status:** ⭐ Golden Recipe

#### 3. MOCVD GaN Epitaxy
- **Process Mode:** MOCVD GaN
- **Target Thickness:** 2000 nm (2 µm)
- **Uniformity:** 97%
- **Process Time:** 100 minutes
- **Temperature:** 1050°C
- **Gases:** TMGa (50 sccm), NH3 (2000 sccm), H2 carrier (8000 sccm)
- **Pressure:** 26,664 Pa (200 Torr)
- **V/III Ratio:** 40:1
- **Tags:** gan, mocvd, epitaxy, led
- **Status:** ✅ Baseline | ⭐ Golden Recipe

---

### Runs (9 total)

**Completed Runs (8):**
- 5x LPCVD Si3N4 runs (LOT-2024-001 through 005)
  - 25 wafers each
  - Duration: ~35 minutes
  - Status: COMPLETED
  - Thickness: ~100 nm

- 3x PECVD SiO2 runs (LOT-PECVD-1 through 3)
  - 50 wafers each
  - Duration: ~15 minutes
  - Status: COMPLETED
  - Thickness: ~500 nm

**In-Progress Run (1):**
- 1x PECVD SiO2 (LOT-CURRENT)
  - 25 wafers
  - Status: PROCESSING
  - Started: 5 minutes ago

---

### SPC Series (2 total)

#### 1. Thickness Control Chart
- **Metric:** thickness_nm
- **Recipe:** LPCVD Si3N4 Standard
- **Chart Type:** X-bar R (subgroup = 5)
- **Control Limits:**
  - UCL: 110 nm
  - CL: 100 nm
  - LCL: 90 nm
- **Specification Limits:**
  - USL: 115 nm
  - LSL: 85 nm

#### 2. Uniformity Control Chart
- **Metric:** uniformity_pct
- **Recipe:** LPCVD Si3N4 Standard
- **Chart Type:** I-MR (Individual & Moving Range)
- **Control Limits:**
  - UCL: 98%
  - CL: 95%
  - LCL: 92%

---

## API Endpoints Available

### Process Modes
```
GET    /api/v1/cvd/process-modes          # List all process modes
POST   /api/v1/cvd/process-modes          # Create new process mode
GET    /api/v1/cvd/process-modes/{id}     # Get specific process mode
PATCH  /api/v1/cvd/process-modes/{id}     # Update process mode
```

### Recipes
```
GET    /api/v1/cvd/recipes                # List all recipes
POST   /api/v1/cvd/recipes                # Create new recipe
GET    /api/v1/cvd/recipes/{id}           # Get specific recipe
PATCH  /api/v1/cvd/recipes/{id}           # Update recipe
```

### Runs
```
GET    /api/v1/cvd/runs                   # List all runs
POST   /api/v1/cvd/runs                   # Create new run
POST   /api/v1/cvd/runs/batch             # Create batch runs
GET    /api/v1/cvd/runs/{id}              # Get specific run
PATCH  /api/v1/cvd/runs/{id}              # Update run status
```

### Telemetry
```
GET    /api/v1/cvd/telemetry/run/{id}    # Get telemetry for run
POST   /api/v1/cvd/telemetry              # Add telemetry point
POST   /api/v1/cvd/telemetry/bulk         # Bulk upload telemetry
WS     /api/v1/cvd/ws/telemetry/{id}     # Real-time telemetry stream
```

### SPC
```
GET    /api/v1/cvd/spc/series             # List SPC series
POST   /api/v1/cvd/spc/series             # Create SPC series
GET    /api/v1/cvd/spc/points/{series_id} # Get SPC points
POST   /api/v1/cvd/spc/points             # Add SPC point
```

### Analytics
```
POST   /api/v1/cvd/analytics              # Query aggregated data
```

---

## Testing the API

### Example: Get All Process Modes

```bash
curl http://localhost:8001/api/v1/cvd/process-modes | jq
```

**Expected Response:**
```json
[
  {
    "id": "uuid",
    "pressure_mode": "LPCVD",
    "energy_mode": "THERMAL",
    "variant": "LPCVD-Si3N4",
    "materials": ["Si3N4", "SiO2", "Poly-Si"],
    "is_active": true,
    ...
  },
  ...
]
```

### Example: Get All Recipes

```bash
curl http://localhost:8001/api/v1/cvd/recipes | jq
```

### Example: Get Active Runs

```bash
curl "http://localhost:8001/api/v1/cvd/runs?status=PROCESSING" | jq
```

---

## Troubleshooting

### Issue: Frontend Still Shows Zeros

**Solution:**
1. Check if seed script ran successfully:
   ```bash
   docker-compose exec analysis python scripts/seed_cvd_data.py
   ```

2. Verify data in database:
   ```bash
   docker-compose exec db psql -U spectra -d spectra -c "SELECT COUNT(*) FROM cvd_process_modes;"
   docker-compose exec db psql -U spectra -d spectra -c "SELECT COUNT(*) FROM cvd_recipes;"
   docker-compose exec db psql -U spectra -d spectra -c "SELECT COUNT(*) FROM cvd_runs;"
   ```

3. Restart frontend:
   ```bash
   docker-compose restart web
   ```

### Issue: API Returns 404

**Solution:**
1. Check if analysis service is running:
   ```bash
   docker-compose ps analysis
   ```

2. Check logs:
   ```bash
   docker-compose logs -f analysis
   ```

3. Verify CVD router is registered:
   ```bash
   curl http://localhost:8001/ | jq '.endpoints'
   # Should show: "cvd": "/api/v1/cvd"
   ```

### Issue: Database Connection Errors

**Solution:**
1. Check database is healthy:
   ```bash
   docker-compose ps db
   docker-compose logs db
   ```

2. Verify connection string:
   ```bash
   docker-compose exec analysis env | grep DATABASE_URL
   # Should be: postgresql+psycopg://spectra:spectra@db:5432/spectra
   ```

3. Restart all services:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

---

## Next Steps

### 1. Explore the Frontend
- Navigate to http://localhost:3012/cvd/workspace
- Browse process modes, recipes, and runs
- Click on individual items to see details

### 2. Use the Recipe Editor
- Go to Recipes tab
- Click "Create Recipe"
- Configure temperature zones, gas flows, and process steps
- Save and launch runs

### 3. Monitor Real-Time Telemetry
- Open an in-progress run
- View live temperature, pressure, and gas flow charts
- Monitor alarms and process stability

### 4. Review SPC Charts
- Go to Analytics tab
- View control charts for thickness and uniformity
- Check process capability (Cpk)
- Identify out-of-control conditions

### 5. Create New Runs
- Use the Run Configuration Wizard
- Select recipe and tool
- Configure wafer lot and IDs
- Launch batch runs

---

## Advanced Usage

### Custom Seed Data

To seed with your own data, modify `services/analysis/scripts/seed_cvd_data.py`:

```python
# Add your custom process mode
custom_pm = CVDProcessMode(
    id=uuid4(),
    organization_id=org_id,
    pressure_mode="PECVD",
    energy_mode="PLASMA",
    # ... your configuration
)
```

### API Integration

Use the TypeScript client for frontend integration:

```typescript
import { cvdApi } from '@/lib/api/cvd';

// Get all recipes
const recipes = await cvdApi.getRecipes({
  organization_id: orgId,
  is_active: true
});

// Create a new run
const run = await cvdApi.createRun({
  recipe_id: recipeId,
  tool_id: toolId,
  wafer_ids: ['W001', 'W002']
});

// Stream real-time telemetry
const ws = cvdApi.connectTelemetryStream(runId);
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Temperature:', data.temperatures);
};
```

---

## Support

- **API Documentation:** http://localhost:8001/docs
- **CVD Platform Guide:** See `CVD_PLATFORM_COMPLETION_SUMMARY.md`
- **Implementation Details:** See `MASTER_IMPLEMENTATION_GUIDE.md`

---

**Status:** ✅ Ready to Use | All integration issues resolved | Sample data available
