# Diffusion Module Integration Mapping

This document maps all files from the Diffusion Module Skeleton Package to their integration locations in the SPECTRA-Lab platform.

## Integration Structure Overview

```
services/analysis/app/
├── simulation/              # NEW: Core simulation modules
│   ├── diffusion/          # Diffusion simulation
│   └── oxidation/          # Oxidation simulation
├── methods/spc/advanced/   # NEW: Advanced SPC methods
├── ml/vm/                  # NEW: Virtual metrology
├── io/                     # NEW: I/O utilities
│   ├── loaders/
│   └── writers/
├── api/v1/simulation/      # NEW: Simulation API endpoints
└── tests/                  # NEW: Test suites
    ├── simulation/
    ├── spc/
    └── vm/
```

## File Mapping

### Core Simulation Modules

| Source File | Target Location | Description |
|------------|-----------------|-------------|
| `fick_fd.py` | `app/simulation/diffusion/fick_fd.py` | Finite difference solver for Fick's second law |
| `massoud.py` | `app/simulation/diffusion/massoud.py` | Massoud diffusion model with clustering |
| `erfc.py` | `app/simulation/diffusion/erfc.py` | Complementary error function solutions |
| `segregation.py` | `app/simulation/diffusion/segregation.py` | Dopant segregation at interfaces |
| `deal_grove.py` | `app/simulation/oxidation/deal_grove.py` | Deal-Grove oxidation model |

### Statistical Process Control (SPC) Modules

| Source File | Target Location | Description |
|------------|-----------------|-------------|
| `cusum.py` | `app/methods/spc/advanced/cusum.py` | CUSUM control charts |
| `ewma.py` | `app/methods/spc/advanced/ewma.py` | EWMA control charts |
| `changepoint.py` | `app/methods/spc/advanced/changepoint.py` | Changepoint detection algorithms |
| `rules.py` | `app/methods/spc/advanced/rules.py` | Western Electric & Nelson rules |

### Machine Learning / Virtual Metrology Modules

| Source File | Target Location | Description |
|------------|-----------------|-------------|
| `vm.py` | `app/ml/vm/vm.py` | Virtual metrology models |
| `forecast.py` | `app/ml/vm/forecast.py` | Time series forecasting |
| `features.py` | `app/ml/vm/features.py` | Feature engineering utilities |

### API Modules

| Source File | Target Location | Description |
|------------|-----------------|-------------|
| `routers.py` | `app/api/v1/simulation/routers.py` | FastAPI simulation endpoints |
| `schemas.py` | `app/api/v1/simulation/schemas.py` | Pydantic request/response models |

### I/O Modules

| Source File | Target Location | Description |
|------------|-----------------|-------------|
| `loaders.py` | `app/io/loaders/loaders.py` | Data loading utilities |
| `writers.py` | `app/io/writers/writers.py` | Data writing utilities |

### Simulation Runner Scripts

| Source File | Target Location | Description |
|------------|-----------------|-------------|
| `run_diffusion_sim.py` | `scripts/simulation/run_diffusion_sim.py` | CLI runner for diffusion simulations |
| `run_oxidation_sim.py` | `scripts/simulation/run_oxidation_sim.py` | CLI runner for oxidation simulations |
| `calibrate.py` | `scripts/simulation/calibrate.py` | Model calibration script |

### Configuration Files

| Source File | Target Location | Description |
|------------|-----------------|-------------|
| `config.py` | `app/simulation/config.py` | Simulation configuration management |
| `pyproject.toml` | Reference for dependencies | Python project metadata |
| `requirements.txt` | Merge into `services/analysis/requirements.txt` | Python dependencies |

### Test Files

| Source File | Target Location | Description |
|------------|-----------------|-------------|
| `conftest.py` | `app/tests/conftest.py` | Pytest configuration and fixtures |
| `test_imports.py` | `app/tests/test_imports.py` | Import validation tests |
| `test_config.py` | `app/tests/simulation/test_config.py` | Configuration tests |
| `test_schemas.py` | `app/tests/simulation/test_schemas.py` | Schema validation tests |

### Documentation Files

| Source File | Purpose |
|------------|---------|
| `README.md` | Reference for module overview |
| `START_HERE.md` | Integration instructions |
| `SESSION_1_COMPLETE.md` | Implementation status reference |
| `SESSION_1_STATUS.md` | Status tracking reference |
| `DELIVERY_SUMMARY.md` | Delivery summary reference |
| `DELIVERY_MANIFEST.md` | File inventory reference |

### Output Directory Structure

| Source Directory | Target Location | Description |
|-----------------|-----------------|-------------|
| `mnt/user-data/outputs/diffusion_oxidation_session1/spc/` | `data/simulation/spc/` | SPC output data |
| `mnt/user-data/outputs/diffusion_oxidation_session1/core/` | `data/simulation/core/` | Core simulation outputs |
| `mnt/user-data/outputs/diffusion_oxidation_session1/io/` | `data/simulation/io/` | I/O test data |
| `mnt/user-data/outputs/diffusion_oxidation_session1/tests/` | `data/simulation/tests/` | Test output data |
| `mnt/user-data/outputs/diffusion_oxidation_session1/ml/` | `data/simulation/ml/` | ML/VM output data |
| `mnt/user-data/outputs/diffusion_oxidation_session1/api/` | `data/simulation/api/` | API test data |

## Integration Steps

### Phase 1: Configuration & Dependencies
1. Review `pyproject.toml` and `requirements.txt` for dependencies
2. Merge dependencies into `services/analysis/requirements.txt`
3. Install dependencies: `pip3 install -r requirements.txt`
4. Integrate `config.py` into `app/simulation/config.py`

### Phase 2: Core Simulation Modules
1. Place diffusion modules in `app/simulation/diffusion/`
2. Place oxidation modules in `app/simulation/oxidation/`
3. Update `__init__.py` files to export module functions
4. Verify imports work correctly

### Phase 3: SPC Modules
1. Place advanced SPC modules in `app/methods/spc/advanced/`
2. Update `__init__.py` to export SPC functions
3. Ensure integration with existing `spc_hub.py`

### Phase 4: ML/VM Modules
1. Place VM modules in `app/ml/vm/`
2. Update `__init__.py` to export VM functions
3. Verify integration with existing ML infrastructure

### Phase 5: API Integration
1. Place API routers and schemas in `app/api/v1/simulation/`
2. Register routers in `app/main.py`
3. Test API endpoints with curl/Postman

### Phase 6: I/O Modules
1. Place loaders in `app/io/loaders/`
2. Place writers in `app/io/writers/`
3. Test data loading and writing functionality

### Phase 7: Simulation Runners
1. Create `scripts/simulation/` directory
2. Place runner scripts
3. Make scripts executable: `chmod +x scripts/simulation/*.py`
4. Test simulation execution

### Phase 8: Testing Infrastructure
1. Place `conftest.py` in `app/tests/`
2. Place test files in appropriate test directories
3. Run test suite: `pytest app/tests/`
4. Verify all tests pass

### Phase 9: Data Directory Setup
1. Create data directories for simulation outputs
2. Set up proper permissions
3. Configure output paths in config files

### Phase 10: Final Integration
1. Update main application to import simulation modules
2. Add simulation endpoints to API documentation
3. Run full integration test suite
4. Update SPECTRA-Lab README with simulation capabilities

## API Integration

### Main Application Update

Add to `app/main.py`:
```python
from app.api.v1.simulation import routers as simulation_router

app.include_router(simulation_router, prefix="/api/v1")
```

Update root endpoint to include simulation:
```python
@app.get("/")
async def root():
    return {
        "service": "analysis",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "automl": "/api/v1/automl",
            "explainability": "/api/v1/explainability",
            "ab_testing": "/api/v1/ab-testing",
            "monitoring": "/api/v1/monitoring",
            "simulation": "/api/v1/simulation"  # NEW
        }
    }
```

## Expected Dependencies

Based on typical diffusion/simulation modules, expected dependencies include:
- `numpy` - Numerical computing
- `scipy` - Scientific computing (for ERFC, optimization)
- `pandas` - Data manipulation
- `matplotlib` - Plotting (for visualization)
- `scikit-learn` - ML utilities
- `pydantic` - Data validation
- `fastapi` - API framework
- `pytest` - Testing framework

## Validation Checklist

- [ ] All modules import without errors
- [ ] All API endpoints respond correctly
- [ ] All tests pass
- [ ] Configuration loads properly
- [ ] Simulation outputs are generated correctly
- [ ] SPC methods integrate with existing spc_hub.py
- [ ] VM models can be trained and used for prediction
- [ ] API documentation is updated
- [ ] Data directories are created with proper permissions
- [ ] Runner scripts execute successfully

## Notes

- This structure maintains separation of concerns while integrating with existing SPECTRA-Lab architecture
- The simulation modules are isolated in their own package for maintainability
- API endpoints follow the existing `/api/v1` pattern
- Tests are organized by module type for clarity
- Data directories follow the existing SPECTRA-Lab data organization pattern

## Status

**Current Status**: Directory structure created, awaiting source files for integration

**Next Step**: Once source files are provided, begin Phase 1 (Configuration & Dependencies)
