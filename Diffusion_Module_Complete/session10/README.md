# Session 10: API Hardening & CLI Tools - PRODUCTION READY

**Status:** ✅ Production Ready
**Date:** November 8, 2025
**Tag:** `diffusion-v10` (pending)

---

## 🎯 Goal

Production-grade API with strong validation and batch CLI tools for diffusion simulation, oxidation simulation, and SPC monitoring.

---

## 📦 Deliverables

### 1. Production-Grade API Schemas (`api/schemas.py` - 500+ lines) ✅ COMPLETE

**Comprehensive Pydantic models with validation:**

**Diffusion Models:**
- `DiffusionRequest` - Full validation with method-specific parameters
- `DiffusionResponse` - Structured output with metrics
- Enums: `DopantType`, `DiffusionMethod`, `SolverType`

**Oxidation Models:**
- `OxidationRequest` - Deal-Grove parameters with bounds
- `OxidationResponse` - Thickness and rate constants
- Enum: `AmbientType`

**SPC Models:**
- `SPCRequest` - Time series with multiple methods
- `SPCResponse` - Violations and change points
- `RuleViolationDetail`, `ChangePointDetail`
- Enums: `SPCMethod`, `SPCRuleType`, `SPCSeverity`

**Virtual Metrology Models:**
- `VMRequest` - Feature dictionary with model selection
- `VMResponse` - Predictions with uncertainty
- Enum: `VMModelType`

**Calibration Models:**
- `CalibrationRequest` - Parameter estimation requests
- `CalibrationResponse` - Parameters with uncertainties
- Enum: `CalibrationMethod`

**Batch Models:**
- `BatchDiffusionRequest` / `BatchDiffusionResponse`
- `BatchOxidationRequest` / `BatchOxidationResponse`

**Common Models:**
- `ErrorResponse`, `ErrorDetail`
- `StatusResponse`

**Key Features:**
- Field validation with `pydantic.Field`
- Custom validators for method-specific params
- Temperature/time/pressure bounds checking
- JSON schema examples for OpenAPI
- Comprehensive type hints

### 2. CLI Tool: batch_diffusion_sim.py ✅ COMPLETE

**Batch diffusion simulator with CSV input/Parquet output.**

**Features:**
- CSV input validation (run_id, dopant, temp, time, method, etc.)
- Support for constant_source and limited_source methods
- ERFC analytical solver and numerical FD solver
- Per-run error handling with status tracking
- Parquet output with schema validation
- Verbose progress reporting
- Summary statistics

**Usage:**
```bash
batch_diffusion_sim.py --input runs.csv --out results.parquet [--method erfc|numerical] [--verbose]
```

**Input CSV Format:**
```csv
run_id,dopant,time_minutes,temp_celsius,method,surface_conc,dose,background
R001,B,30,1000,constant_source,1e19,,1e15
R002,P,60,950,limited_source,,1e14,1e15
```

**Output Columns:**
- run_id, dopant, temp_celsius, time_minutes, method, solver
- junction_depth_nm, sheet_resistance_ohm_sq, peak_concentration
- status (SUCCESS/FAILED), error_message

### 3. CLI Tool: batch_oxidation_sim.py ✅ COMPLETE

**Batch oxidation simulator using Deal-Grove model.**

**Features:**
- CSV input validation (recipe_id, temp, time, ambient, pressure)
- Deal-Grove model for dry and wet oxidation
- Optional initial oxide thickness
- Growth rate calculations
- Parquet output with full metrics
- Summary statistics for successful runs

**Usage:**
```bash
batch_oxidation_sim.py --input recipes.csv --out results.parquet [--verbose]
```

**Input CSV Format:**
```csv
recipe_id,temp_celsius,time_hours,ambient,pressure,initial_thickness_nm
OX001,1000,2.0,dry,1.0,0
OX002,1100,1.0,wet,1.0,5.0
```

**Output Columns:**
- recipe_id, temp_celsius, time_hours, ambient, pressure
- initial_thickness_nm, final_thickness_nm, growth_thickness_nm
- growth_rate_nm_hr, B_parabolic_nm2_hr, A_linear_nm
- status, error_message

### 4. CLI Tool: spc_watch.py ✅ COMPLETE

**SPC monitoring for KPI time series with multiple methods.**

**Features:**
- CSV time series input (timestamp, value)
- Multiple SPC methods: rules, EWMA, CUSUM, BOCPD
- Western Electric & Nelson rules (8 rules)
- JSON output report with structured violations
- Change point detection
- Summary statistics

**Usage:**
```bash
spc_watch.py --series kpi.csv --report spc.json [--methods all|rules|ewma|cusum|bocpd] [--verbose]
```

**Input CSV Format:**
```csv
timestamp,value
2025-01-01T00:00:00,100.5
2025-01-01T01:00:00,102.3
```

**Output JSON Structure:**
```json
{
  "metadata": {
    "generated_at": "2025-01-01T00:00:00Z",
    "methods_applied": ["rules", "ewma", "cusum", "bocpd"]
  },
  "summary": {
    "n_observations": 100,
    "total_violations": 5,
    "total_changepoints": 2,
    "mean": 100.5,
    "std": 5.2
  },
  "results": {
    "spc_rules": { "n_violations": 3, "violations": [...] },
    "ewma": { "n_violations": 1, "violations": [...] },
    "cusum": { "n_violations": 1, "violations": [...] },
    "bocpd": { "n_changepoints": 2, "changepoints": [...] }
  }
}
```

### 5. E2E Tests (`tests/`) ✅ COMPLETE

**Comprehensive end-to-end testing:**

**test_cli_e2e.py:**
- Tests for all three CLI tools
- Fixture-based CSV generation
- Output validation (Parquet, JSON)
- Integration tests combining all tools
- ~300 lines of test code

**test_schemas.py:**
- Validation tests for all Pydantic models
- Boundary condition testing
- Enum validation
- Error case testing
- JSON schema example validation
- ~400 lines of test code

**Test Coverage:**
- ✅ Diffusion CLI with ERFC and numerical solvers
- ✅ Oxidation CLI with dry/wet oxidation
- ✅ SPC CLI with all four methods
- ✅ Schema validation for all models
- ✅ Integration workflow tests

---

## 📊 Stats

**Lines of Code:** 1500+ total
- api/schemas.py: 500+ lines
- scripts/batch_diffusion_sim.py: 314 lines
- scripts/batch_oxidation_sim.py: 280 lines
- scripts/spc_watch.py: 400 lines
- tests/test_cli_e2e.py: 300+ lines
- tests/test_schemas.py: 400+ lines

**Files Created:** 9 files in session10/
**CLI Tools:** 3 (diffusion, oxidation, SPC)
**API Models:** 20+ Pydantic schemas
**Tests:** 50+ test cases
**Production Status:** ✅ Complete and Production Ready

---

## ✅ What's Complete

1. ✅ **Production API Schemas**
   - Comprehensive Pydantic models
   - Field validation with bounds
   - Custom validators
   - JSON schema examples
   - Error handling models

2. ✅ **Batch Diffusion CLI**
   - CSV input validation
   - ERFC and numerical solvers
   - Parquet output
   - Error handling per run
   - Summary statistics

3. ✅ **Batch Oxidation CLI**
   - Deal-Grove model integration
   - Dry and wet oxidation
   - Initial oxide support
   - Growth rate calculations
   - Comprehensive output

4. ✅ **SPC Watch CLI**
   - Multiple SPC methods
   - Western Electric/Nelson rules
   - EWMA, CUSUM, BOCPD
   - JSON report output
   - Change point detection

5. ✅ **E2E Test Suite**
   - CLI integration tests
   - Schema validation tests
   - Fixture-based testing
   - Full workflow coverage
   - 50+ test cases

6. ✅ **Documentation**
   - Comprehensive README
   - Usage examples
   - Input/output specifications
   - CLI help messages

---

## 🔄 Integration Points

### With Session 2 (ERFC Diffusion)
- batch_diffusion_sim.py uses `constant_source_profile()` and `limited_source_profile()`
- Imports from session2.erfc

### With Session 3 (Numerical Solver)
- batch_diffusion_sim.py uses `solve_diffusion_1d()`
- Numerical solver backend option

### With Session 4 (Deal-Grove)
- batch_oxidation_sim.py uses `thickness_at_time()`, `growth_rate()`, `get_rate_constants()`
- Imports from session4.deal_grove

### With Session 7 (SPC)
- spc_watch.py uses all SPC modules
- Imports from session7.spc

### With Session 9 (Calibration)
- API schemas support calibration requests/responses
- Ready for FastAPI endpoint integration

---

## 💡 CLI Usage Examples

### Example 1: Batch Diffusion Simulation

```bash
# Create input CSV
cat > runs.csv << EOF
run_id,dopant,time_minutes,temp_celsius,method,surface_conc,dose,background
R001,B,30,1000,constant_source,1e19,,1e15
R002,P,60,950,limited_source,,1e14,1e15
R003,As,45,1050,constant_source,5e18,,1e15
EOF

# Run simulation
python3 session10/scripts/batch_diffusion_sim.py \
  --input runs.csv \
  --out results.parquet \
  --method erfc \
  --verbose

# Results saved to results.parquet
```

### Example 2: Batch Oxidation Simulation

```bash
# Create input CSV
cat > recipes.csv << EOF
recipe_id,temp_celsius,time_hours,ambient
OX001,1000,2.0,dry
OX002,1100,1.0,wet
OX003,950,3.0,dry
EOF

# Run simulation
python3 session10/scripts/batch_oxidation_sim.py \
  --input recipes.csv \
  --out ox_results.parquet \
  --verbose

# Results include thickness, growth rate, B/A parameters
```

### Example 3: SPC Monitoring

```bash
# Create KPI time series CSV
cat > kpi.csv << EOF
timestamp,value
2025-01-01T00:00:00,100.5
2025-01-01T01:00:00,102.3
2025-01-01T02:00:00,98.7
...
EOF

# Run SPC analysis
python3 session10/scripts/spc_watch.py \
  --series kpi.csv \
  --report spc_report.json \
  --methods all \
  --verbose

# JSON report includes violations and change points
```

---

## 🧪 Running Tests

**Run all E2E tests:**
```bash
cd Diffusion_Module_Complete
pytest session10/tests/ -v
```

**Run CLI tests only:**
```bash
pytest session10/tests/test_cli_e2e.py -v
```

**Run schema tests only:**
```bash
pytest session10/tests/test_schemas.py -v
```

**Run specific test class:**
```bash
pytest session10/tests/test_cli_e2e.py::TestBatchDiffusionSim -v
```

---

## 📋 API Schema Examples

### Diffusion Request Example

```json
{
  "dopant": "boron",
  "temp_celsius": 1000,
  "time_minutes": 30,
  "method": "constant_source",
  "surface_conc": 1e19,
  "background": 1e15,
  "depth_nm": [0, 50, 100, 150, 200, 250, 300],
  "solver": "erfc"
}
```

### Oxidation Request Example

```json
{
  "temp_celsius": 1000,
  "time_hours": 2.0,
  "ambient": "dry",
  "pressure": 1.0,
  "initial_thickness_nm": 5.0
}
```

### SPC Request Example

```json
{
  "data": [
    {"timestamp": "2025-01-01T00:00:00", "value": 100.0},
    {"timestamp": "2025-01-01T01:00:00", "value": 102.0},
    {"timestamp": "2025-01-01T02:00:00", "value": 98.5}
  ],
  "methods": ["rules", "ewma"],
  "enabled_rules": ["RULE_1", "RULE_2"]
}
```

---

## 🚀 Next Steps (Optional)

1. **FastAPI Integration**
   - Create FastAPI app using schemas
   - Implement all endpoints
   - Add authentication/authorization
   - Deploy API server

2. **Additional CLI Features**
   - Progress bars for long runs
   - Parallel processing for batch operations
   - Resume capability for interrupted runs
   - Advanced output formats (HDF5, CSV)

3. **Enhanced Validation**
   - Cross-field validation
   - Domain-specific constraints
   - Warning vs. error levels
   - Validation reports

4. **Documentation**
   - OpenAPI/Swagger UI
   - API client examples
   - Jupyter notebooks
   - Video tutorials

---

## 📚 File Structure

```
session10/
├── README.md                    # This file
├── __init__.py
├── api/
│   ├── __init__.py
│   └── schemas.py               # Pydantic models
├── scripts/
│   ├── batch_diffusion_sim.py   # Diffusion CLI
│   ├── batch_oxidation_sim.py   # Oxidation CLI
│   └── spc_watch.py             # SPC CLI
└── tests/
    ├── __init__.py
    ├── test_cli_e2e.py          # CLI integration tests
    └── test_schemas.py          # Schema validation tests
```

---

**Status:** ✅ PRODUCTION READY - ALL COMPONENTS COMPLETE

**Lines of Code:** 1500+
**CLI Tools:** 3 (diffusion, oxidation, SPC)
**API Models:** 20+ Pydantic schemas
**Tests:** 50+ test cases
**Dependencies:** pandas, numpy, pydantic, pyarrow, pytest

**Ready for:** Production deployment, git tag `diffusion-v10`
