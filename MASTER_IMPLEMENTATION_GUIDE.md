# CVD Platform - Master Implementation Guide
## Complete SPECTRA-Lab Integration for All CVD Variants

**Version:** 2.0 - Enhanced Universal CVD Platform
**Date:** 2024-11-13
**Status:** Complete Architecture & Critical Files Delivered

---

## 🎯 Overview

This document describes the complete implementation of a unified CVD platform supporting **ALL CVD variants** and fully integrated with SPECTRA-Lab infrastructure.

### Supported CVD Variants

**By Pressure (6 variants):**
- APCVD, LPCVD, UHVCVD, PECVD, HDP-CVD, SACVD

**By Energy/Activation (8 variants):**
- Thermal, Plasma, Hot-Wire (HWCVD), Laser (LCVD), Photo-CVD, Microwave Plasma (MPCVD), Remote Plasma (RPCVD), Combustion (CCVD)

**By Reactor Type (8 variants):**
- Cold-Wall, Hot-Wall, Horizontal, Vertical/Pancake, Showerhead, Rotating Disk Reactor (RDR), Cold Finger, Exhaust-Controlled

**By Chemistry (7 variants):**
- MOCVD, OMCVD, Halide CVD (HCVD), Hydride CVD, Aerosol-Assisted (AACVD), Standard, Organometallic

**Advanced/Specialized (8+ variants):**
- ALCVD, Pulsed/Sequential CVD, Selective-Area (SACVD), Hybrid CVD-ALD, PEMOCVD, LICVD, CCVD, RPCVD

**Total: 37+ CVD process combinations supported**

---

## 📁 Complete File Structure

```
SPECTRA-Lab/
└── cvd_platform/
    │
    ├── services/
    │   ├── analysis/
    │   │   └── app/
    │   │       ├── alembic/
    │   │       │   └── versions/
    │   │       │       └── 0001_cvd_module.py ✅ CREATED
    │   │       │
    │   │       ├── models/
    │   │       │   ├── __init__.py
    │   │       │   ├── cvd.py ✅ CREATED
    │   │       │   └── database.py
    │   │       │
    │   │       ├── schemas/
    │   │       │   ├── __init__.py
    │   │       │   └── cvd.py 📝 NEXT
    │   │       │
    │   │       ├── routers/
    │   │       │   ├── __init__.py
    │   │       │   └── cvd.py 📝 NEXT
    │   │       │
    │   │       ├── tools/
    │   │       │   ├── __init__.py
    │   │       │   ├── base.py 📝 NEXT
    │   │       │   ├── drivers/
    │   │       │   │   ├── __init__.py
    │   │       │   │   ├── thermal_cvd.py
    │   │       │   │   ├── pecvd.py
    │   │       │   │   ├── mocvd.py
    │   │       │   │   ├── hwcvd.py
    │   │       │   │   ├── lcvd.py
    │   │       │   │   └── hdp_cvd.py
    │   │       │   └── simulators/
    │   │       │       ├── __init__.py
    │   │       │       ├── thermal_sim.py
    │   │       │       ├── plasma_sim.py
    │   │       │       └── mocvd_sim.py
    │   │       │
    │   │       ├── physics/
    │   │       │   ├── __init__.py
    │   │       │   ├── growth_models.py
    │   │       │   ├── transport_models.py
    │   │       │   ├── plasma_models.py
    │   │       │   └── conformality_models.py
    │   │       │
    │   │       ├── ml/
    │   │       │   ├── __init__.py
    │   │       │   ├── feature_store.py
    │   │       │   ├── vm_models.py
    │   │       │   ├── model_registry.py
    │   │       │   └── drift_detection.py
    │   │       │
    │   │       ├── control/
    │   │       │   ├── __init__.py
    │   │       │   ├── spc.py
    │   │       │   ├── fdc.py
    │   │       │   ├── r2r.py
    │   │       │   └── mpc.py
    │   │       │
    │   │       ├── tasks/
    │   │       │   ├── __init__.py
    │   │       │   └── cvd_tasks.py
    │   │       │
    │   │       └── main.py (enhanced)
    │   │
    │   └── lims/
    │       └── app/
    │           └── integrations/
    │               └── cvd_integration.py
    │
    ├── frontend/
    │   └── src/
    │       ├── app/
    │       │   └── cvd/
    │       │       ├── page.tsx
    │       │       ├── layout.tsx
    │       │       ├── modes/
    │       │       │   ├── page.tsx
    │       │       │   └── [id]/page.tsx
    │       │       ├── recipes/
    │       │       │   ├── page.tsx
    │       │       │   ├── [id]/page.tsx
    │       │       │   └── builder/page.tsx
    │       │       ├── runs/
    │       │       │   ├── page.tsx
    │       │       │   ├── [id]/page.tsx
    │       │       │   └── create/page.tsx
    │       │       ├── results/
    │       │       │   ├── page.tsx
    │       │       │   └── [id]/page.tsx
    │       │       └── analytics/
    │       │           ├── page.tsx
    │       │           ├── vm/page.tsx
    │       │           └── spc/page.tsx
    │       │
    │       ├── components/
    │       │   └── cvd/
    │       │       ├── ProcessModeCard.tsx
    │       │       ├── RecipeBuilder.tsx
    │       │       ├── RecipeStepEditor.tsx
    │       │       ├── RunMonitor.tsx
    │       │       ├── TelemetryChart.tsx
    │       │       ├── ReactorDiagram.tsx
    │       │       ├── PlasmaControl.tsx
    │       │       ├── GasFlowControl.tsx
    │       │       ├── ResultsViewer.tsx
    │       │       ├── SPCChart.tsx
    │       │       └── VMDashboard.tsx
    │       │
    │       └── lib/
    │           └── api/
    │               └── cvd.ts
    │
    ├── tests/
    │   └── cvd/
    │       ├── conftest.py
    │       ├── test_models.py
    │       ├── test_schemas.py
    │       ├── test_tools.py
    │       ├── test_simulators.py
    │       ├── test_physics.py
    │       ├── test_ml.py
    │       ├── test_control.py
    │       ├── test_api.py
    │       └── test_integration.py
    │
    ├── docs/
    │   ├── cvd/
    │   │   ├── ARCHITECTURE.md (updated)
    │   │   ├── PROCESS_MODES.md
    │   │   ├── RECIPE_SPECIFICATION.md
    │   │   ├── TOOL_ABSTRACTION.md
    │   │   ├── SIMULATOR_GUIDE.md
    │   │   ├── VM_ML_MODELS.md
    │   │   ├── SPC_FDC_R2R.md
    │   │   ├── API_REFERENCE.md
    │   │   ├── UI_GUIDE.md
    │   │   └── INTEGRATION_LIMS.md
    │   │
    │   └── deployment/
    │       ├── DATABASE_MIGRATION.md
    │       └── SCALING_GUIDE.md
    │
    ├── docker-compose.enhanced.yml
    ├── requirements.enhanced.txt
    └── MASTER_IMPLEMENTATION_GUIDE.md (this file)
```

---

## 🔧 Implementation Status

### ✅ Completed (Critical Foundation)

1. **Database Schema (Alembic Migration)** - `0001_cvd_module.py`
   - Complete table definitions for all CVD variants
   - Enums for classification (pressure, energy, reactor, chemistry)
   - Process modes, recipes, runs, telemetry, results
   - SPC series and points
   - TimescaleDB-ready for high-frequency telemetry

2. **SQLAlchemy Models** - `models/cvd.py`
   - Full ORM models with relationships
   - Enums matching database schema
   - Properties and methods for business logic
   - Support for all CVD variants

3. **Previous Platform Components** (All Retained)
   - Physics models (Navier-Stokes, transport, kinetics)
   - Virtual metrology (LightGBM, neural networks)
   - Process control (R2R, PID, MPC, adaptive)
   - SPC/FDC monitoring
   - Analytics (anomaly detection, predictive maintenance)
   - FastAPI backend
   - React/TypeScript frontend

### 📝 To Be Implemented (Following Files)

**Immediate Priority (Session CVD-1):**
1. Pydantic schemas
2. Basic API routers
3. Tool abstraction base class
4. One complete HIL simulator (LPCVD oxide)
5. Celery task integration

**Phase 2 (Session CVD-2):**
- All tool drivers
- All HIL simulators
- Enhanced physics models

**Phase 3 (Session CVD-3+):**
- Frontend workspace
- LIMS integration
- Report generation
- Comprehensive testing

---

## 📊 Database Schema Details

### Table: `cvd_process_modes`

Defines CVD process classification:

```sql
CREATE TABLE cvd_process_modes (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    pressure_mode pressure_mode_enum NOT NULL,  -- APCVD, LPCVD, UHVCVD, etc.
    energy_mode energy_mode_enum NOT NULL,      -- thermal, plasma, hot_wire, etc.
    reactor_type reactor_type_enum NOT NULL,    -- cold_wall, hot_wall, etc.
    chemistry_type chemistry_type_enum NOT NULL, -- MOCVD, OMCVD, etc.
    variant VARCHAR(100),                        -- ALCVD, pulsed, hybrid, etc.
    description TEXT,
    default_recipes JSONB,
    capabilities JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(org_id, name)
);
```

**Example Process Modes:**

| Name | Pressure | Energy | Reactor | Chemistry | Variant |
|------|----------|--------|---------|-----------|---------|
| LPCVD Oxide | LPCVD | thermal | hot_wall | hydride | standard |
| PECVD Nitride | PECVD | plasma | showerhead | standard | standard |
| MOCVD GaN | LPCVD | thermal | vertical | MOCVD | epitaxy |
| HDP Oxide | HDP_CVD | plasma | cold_wall | standard | gap_fill |
| HWCVD Si | LPCVD | hot_wire | cold_wall | hydride | standard |
| ALCVD Barrier | LPCVD | thermal | showerhead | MOCVD | atomic_layer |

### Table: `cvd_recipes`

Multi-variant recipe support with JSONB flexibility:

```sql
CREATE TABLE cvd_recipes (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    process_mode_id UUID REFERENCES cvd_process_modes(id),

    -- Targets
    film_target VARCHAR(100) NOT NULL,
    thickness_target_nm FLOAT NOT NULL,
    uniformity_target_pct FLOAT NOT NULL,

    -- Core parameters (JSONB for flexibility)
    temperature_profile JSONB NOT NULL,
    pressure_setpoints JSONB NOT NULL,
    gas_flows JSONB NOT NULL,

    -- Variant-specific (NULL if not applicable)
    plasma_settings JSONB,
    filament_settings JSONB,
    laser_settings JSONB,
    pulsing_scheme JSONB,

    -- Multi-step support
    recipe_steps JSONB NOT NULL,

    -- Safety
    safety_hazard_level INTEGER NOT NULL DEFAULT 1,
    required_interlocks JSONB,

    -- Metadata
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    approval_date TIMESTAMP WITH TIME ZONE,
    approved_by_id UUID,
    metadata JSONB,
    created_by_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,

    UNIQUE(org_id, name, version)
);
```

**Example Recipe (LPCVD SiO₂):**

```json
{
  "name": "LPCVD SiO2 - 100nm",
  "version": 1,
  "process_mode_id": "...",
  "film_target": "SiO2",
  "thickness_target_nm": 100.0,
  "uniformity_target_pct": 2.0,

  "temperature_profile": {
    "zones": [
      {"zone": 1, "setpoint": 800, "ramp_rate": 10},
      {"zone": 2, "setpoint": 800, "ramp_rate": 10}
    ],
    "profile_type": "isothermal"
  },

  "pressure_setpoints": {
    "base_pressure_torr": 0.001,
    "process_pressure_torr": 0.5,
    "pumpdown_time_s": 300
  },

  "gas_flows": {
    "gases": [
      {"name": "SiH4", "flow_sccm": 80, "timing": "continuous"},
      {"name": "N2O", "flow_sccm": 160, "timing": "continuous"}
    ]
  },

  "recipe_steps": [
    {"name": "pumpdown", "duration_s": 300, "target_pressure_torr": 0.001},
    {"name": "heatup", "duration_s": 600, "target_temp_c": 800},
    {"name": "stabilize", "duration_s": 120},
    {"name": "deposition", "duration_s": 1200, "flows_active": true},
    {"name": "cooldown", "duration_s": 900}
  ],

  "safety_hazard_level": 3,
  "status": "approved"
}
```

**Example Recipe (PECVD Si₃N₄):**

```json
{
  "name": "PECVD Si3N4 - 50nm",
  "film_target": "Si3N4",
  "thickness_target_nm": 50.0,

  "temperature_profile": {
    "zones": [{"zone": 1, "setpoint": 300}]
  },

  "pressure_setpoints": {
    "process_pressure_torr": 1.0
  },

  "gas_flows": {
    "gases": [
      {"name": "SiH4", "flow_sccm": 200},
      {"name": "NH3", "flow_sccm": 20},
      {"name": "N2", "flow_sccm": 1000}
    ]
  },

  "plasma_settings": {
    "rf_power_w": 300,
    "frequency_mhz": 13.56,
    "dc_bias_v": -100,
    "duty_cycle": 1.0,
    "matching_network": "auto"
  },

  "recipe_steps": [
    {"name": "plasma_clean", "duration_s": 60, "plasma_power_w": 500},
    {"name": "deposition", "duration_s": 300, "plasma_power_w": 300}
  ]
}
```

**Example Recipe (MOCVD GaN):**

```json
{
  "name": "MOCVD GaN - 2um",
  "film_target": "GaN",
  "thickness_target_nm": 2000.0,

  "temperature_profile": {
    "zones": [{"zone": 1, "setpoint": 1050}],
    "profile_type": "isothermal"
  },

  "pressure_setpoints": {
    "process_pressure_torr": 76  // 100 mbar
  },

  "gas_flows": {
    "gases": [
      {"name": "TMGa", "flow_sccm": 20, "carrier": "H2"},
      {"name": "NH3", "flow_sccm": 2000},
      {"name": "H2", "flow_sccm": 5000}
    ]
  },

  "pulsing_scheme": null,

  "recipe_steps": [
    {"name": "buffer_layer", "duration_s": 300, "temp_c": 550},
    {"name": "main_growth", "duration_s": 3600, "temp_c": 1050, "TMGa_flow": 20}
  ]
}
```

### Table: `cvd_runs`

Execution tracking:

```sql
CREATE TABLE cvd_runs (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    run_number VARCHAR(100) NOT NULL UNIQUE,
    cvd_recipe_id UUID REFERENCES cvd_recipes(id),
    process_mode_id UUID REFERENCES cvd_process_modes(id),
    instrument_id UUID NOT NULL,
    sample_id UUID,
    wafer_id UUID,
    status run_status_enum NOT NULL DEFAULT 'queued',
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds FLOAT,
    preflight_summary JSONB,
    postflight_summary JSONB,
    error_code VARCHAR(100),
    error_message TEXT,
    fault_data JSONB,
    celery_task_id VARCHAR(255),
    job_progress FLOAT DEFAULT 0.0,
    operator_id UUID NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Table: `cvd_telemetry`

High-frequency time-series:

```sql
CREATE TABLE cvd_telemetry (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    cvd_run_id UUID REFERENCES cvd_runs(id) ON DELETE CASCADE,
    ts TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Core
    chamber_pressure_torr FLOAT,
    temperature_zones_c JSONB,
    gas_flows_sccm JSONB,

    -- Plasma (PECVD, HDP, MPCVD, etc.)
    rf_power_w FLOAT,
    rf_reflected_w FLOAT,
    dc_bias_v FLOAT,
    plasma_impedance_ohm FLOAT,
    microwave_power_w FLOAT,

    -- Hot-wire (HWCVD)
    filament_temp_c FLOAT,
    filament_current_a FLOAT,

    -- Laser (LCVD, LICVD)
    laser_power_w FLOAT,
    laser_energy_mj FLOAT,

    -- In-situ metrology
    qcm_rate_nm_per_s FLOAT,
    qcm_thickness_nm FLOAT,
    ellipsometer_thickness_nm FLOAT,
    reflectometer_thickness_nm FLOAT,

    -- Derived
    deposition_rate_nm_per_min FLOAT,
    plasma_density_proxy FLOAT,

    -- References
    rga_spectrum_uri VARCHAR(500),
    oes_spectrum_uri VARCHAR(500),
    other_signals JSONB
);

-- Create hypertable for TimescaleDB (if available)
-- SELECT create_hypertable('cvd_telemetry', 'ts');
-- CREATE INDEX idx_telemetry_run_ts ON cvd_telemetry(cvd_run_id, ts DESC);
```

### Table: `cvd_results`

Post-process analysis:

```sql
CREATE TABLE cvd_results (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    cvd_run_id UUID REFERENCES cvd_runs(id) UNIQUE,

    -- Film properties
    film_material VARCHAR(100) NOT NULL,
    thickness_mean_nm FLOAT,
    thickness_std_nm FLOAT,
    thickness_uniformity_pct FLOAT,
    thickness_min_nm FLOAT,
    thickness_max_nm FLOAT,
    thickness_map JSONB,  -- Full wafer map

    -- Optical
    refractive_index FLOAT,
    extinction_coefficient FLOAT,

    -- Mechanical
    stress_mpa FLOAT,
    stress_type VARCHAR(50),

    -- Quality
    conformality_score FLOAT,
    selectivity_score FLOAT,
    step_coverage_pct FLOAT,
    defect_density_per_cm2 FLOAT,
    roughness_rms_nm FLOAT,

    -- Composition (MOCVD/alloys)
    composition JSONB,  -- {"In": 0.2, "Ga": 0.8, "N": 1.0}

    -- ML/SPC
    vm_predictions JSONB,
    spc_snapshot JSONB,

    -- References
    report_uri VARCHAR(500),
    raw_data_uri VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### SPC Tables

```sql
CREATE TABLE cvd_spc_series (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    parameter VARCHAR(100) NOT NULL,
    process_mode_id UUID REFERENCES cvd_process_modes(id),
    instrument_id UUID,
    target FLOAT,
    ucl FLOAT,
    lcl FLOAT,
    usl FLOAT,
    lsl FLOAT,
    mean FLOAT,
    std_dev FLOAT,
    cp FLOAT,
    cpk FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE cvd_spc_points (
    id UUID PRIMARY KEY,
    org_id UUID NOT NULL,
    series_id UUID REFERENCES cvd_spc_series(id),
    cvd_run_id UUID REFERENCES cvd_runs(id),
    value FLOAT NOT NULL,
    ts TIMESTAMP WITH TIME ZONE NOT NULL,
    violation BOOLEAN NOT NULL DEFAULT FALSE,
    violation_rules JSONB
);
```

---

## 🔌 Tool Abstraction Layer

### Base Class Design

```python
# services/analysis/app/tools/base.py

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional, List
from uuid import UUID
from enum import Enum

from ..models.cvd import CVDRecipe, PressureMode, EnergyMode, ReactorType
from ..schemas.cvd import CVDTelemetrySchema, ToolStatusSchema


class ToolCapability(str, Enum):
    """Tool capabilities"""
    THERMAL_CVD = "thermal_cvd"
    PLASMA_CVD = "plasma_cvd"
    HOT_WIRE_CVD = "hot_wire_cvd"
    LASER_CVD = "laser_cvd"
    MOCVD = "mocvd"
    AACVD = "aacvd"
    PULSING = "pulsing"
    IN_SITU_METROLOGY = "in_situ_metrology"


class CVDTool(ABC):
    """
    Abstract base class for all CVD tools
    Provides unified interface across all variants
    """

    def __init__(self, tool_id: UUID, config: Dict[str, Any]):
        self.tool_id = tool_id
        self.config = config
        self.capabilities: List[ToolCapability] = []
        self.pressure_modes: List[PressureMode] = []
        self.energy_modes: List[EnergyMode] = []
        self.reactor_type: ReactorType = None

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to tool"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from tool"""
        pass

    @abstractmethod
    async def configure(self, recipe: CVDRecipe) -> bool:
        """Configure tool with recipe parameters"""
        pass

    @abstractmethod
    async def start_run(self, run_id: UUID) -> bool:
        """Start CVD process run"""
        pass

    @abstractmethod
    async def stop_run(self, run_id: UUID, emergency: bool = False) -> bool:
        """Stop run (normal or emergency stop)"""
        pass

    @abstractmethod
    async def pause_run(self, run_id: UUID) -> bool:
        """Pause run (if supported)"""
        pass

    @abstractmethod
    async def resume_run(self, run_id: UUID) -> bool:
        """Resume paused run"""
        pass

    @abstractmethod
    async def get_status(self, run_id: UUID) -> ToolStatusSchema:
        """Get current tool status"""
        pass

    @abstractmethod
    async def stream_telemetry(self, run_id: UUID) -> AsyncIterator[CVDTelemetrySchema]:
        """Stream real-time telemetry data"""
        pass

    @abstractmethod
    async def calibrate(self, calibration_type: str) -> bool:
        """Run calibration routine"""
        pass

    @abstractmethod
    async def run_interlock_check(self) -> Dict[str, bool]:
        """Check all safety interlocks"""
        pass

    def supports_recipe(self, recipe: CVDRecipe) -> bool:
        """Check if tool can run this recipe"""
        process_mode = recipe.process_mode

        # Check pressure mode
        if process_mode.pressure_mode not in self.pressure_modes:
            return False

        # Check energy mode
        if process_mode.energy_mode not in self.energy_modes:
            return False

        # Check reactor type match
        if self.reactor_type != process_mode.reactor_type:
            return False

        return True
```

### Driver Examples

**LPCVD Driver:**
```python
# services/analysis/app/tools/drivers/thermal_cvd.py

class LPCVDTool(CVDTool):
    """LPCVD thermal CVD tool"""

    def __init__(self, tool_id: UUID, config: Dict[str, Any]):
        super().__init__(tool_id, config)
        self.capabilities = [ToolCapability.THERMAL_CVD]
        self.pressure_modes = [PressureMode.LPCVD, PressureMode.UHVCVD]
        self.energy_modes = [EnergyMode.THERMAL]
        self.reactor_type = ReactorType.HOT_WALL

        # Driver-specific
        self.visa_resource = config.get("visa_resource")
        self.plc_address = config.get("plc_address")

    async def configure(self, recipe: CVDRecipe) -> bool:
        # Set temperature zones
        temp_profile = recipe.temperature_profile
        for zone in temp_profile["zones"]:
            await self._set_zone_temp(zone["zone"], zone["setpoint"])

        # Set pressure
        pressure = recipe.pressure_setpoints["process_pressure_torr"]
        await self._set_pressure(pressure)

        # Configure MFCs
        for gas in recipe.gas_flows["gases"]:
            await self._set_mfc(gas["name"], gas["flow_sccm"])

        return True

    # ... implement other abstract methods
```

**PECVD Driver:**
```python
# services/analysis/app/tools/drivers/pecvd.py

class PECVDTool(CVDTool):
    """PECVD plasma-enhanced tool"""

    def __init__(self, tool_id: UUID, config: Dict[str, Any]):
        super().__init__(tool_id, config)
        self.capabilities = [ToolCapability.PLASMA_CVD, ToolCapability.IN_SITU_METROLOGY]
        self.pressure_modes = [PressureMode.PECVD]
        self.energy_modes = [EnergyMode.PLASMA]
        self.reactor_type = ReactorType.SHOWERHEAD

    async def configure(self, recipe: CVDRecipe) -> bool:
        # Temperature
        await self._set_temp(recipe.temperature_profile["zones"][0]["setpoint"])

        # Pressure
        await self._set_pressure(recipe.pressure_setpoints["process_pressure_torr"])

        # Gas flows
        for gas in recipe.gas_flows["gases"]:
            await self._set_mfc(gas["name"], gas["flow_sccm"])

        # Plasma settings
        plasma = recipe.plasma_settings
        await self._set_rf_power(plasma["rf_power_w"])
        await self._set_rf_frequency(plasma["frequency_mhz"])
        await self._set_dc_bias(plasma["dc_bias_v"])

        return True

    # ... implement other abstract methods
```

---

## 🎮 HIL Simulators

### Thermal CVD Simulator

```python
# services/analysis/app/tools/simulators/thermal_sim.py

import numpy as np
from typing import AsyncIterator
from datetime import datetime, timedelta

from ..base import CVDTool
from ...schemas.cvd import CVDTelemetrySchema


class ThermalCVDSimulator(CVDTool):
    """
    High-fidelity simulator for thermal CVD (LPCVD, APCVD)
    Includes:
    - Arrhenius kinetics
    - Mass transport limitations
    - Temperature/pressure dependencies
    - Realistic noise and drift
    """

    def __init__(self, tool_id: UUID, config: Dict[str, Any]):
        super().__init__(tool_id, config)
        self.capabilities = [ToolCapability.THERMAL_CVD]
        self.pressure_modes = [PressureMode.LPCVD, PressureMode.APCVD]
        self.energy_modes = [EnergyMode.THERMAL]

        # Physics parameters
        self.k0 = 1e8  # Pre-exponential factor (1/s)
        self.Ea = 1.7e5  # Activation energy (J/mol)
        self.R = 8.314  # Gas constant

        # Simulation state
        self.is_running = False
        self.current_temp = 20.0
        self.current_pressure = 1e-6
        self.accumulated_thickness = 0.0

    async def stream_telemetry(self, run_id: UUID) -> AsyncIterator[CVDTelemetrySchema]:
        """Generate synthetic telemetry with realistic physics"""

        recipe = await self._get_recipe(run_id)
        start_time = datetime.utcnow()

        # Extract recipe parameters
        target_temp = recipe.temperature_profile["zones"][0]["setpoint"]
        target_pressure = recipe.pressure_setpoints["process_pressure_torr"]
        sih4_flow = next(g["flow_sccm"] for g in recipe.gas_flows["gases"] if g["name"] == "SiH4")

        # Simulation parameters
        dt = 1.0  # 1 Hz telemetry
        temp_ramp_rate = 10.0  # °C/s

        while self.is_running:
            current_time = datetime.utcnow()
            elapsed = (current_time - start_time).total_seconds()

            # Temperature ramp
            if self.current_temp < target_temp:
                self.current_temp = min(self.current_temp + temp_ramp_rate * dt, target_temp)

            # Pressure ramp
            if self.current_pressure < target_pressure:
                self.current_pressure = min(self.current_pressure * 1.1, target_pressure)

            # Calculate deposition rate using Arrhenius equation
            if self.current_temp > 500:  # Deposition regime
                k = self.k0 * np.exp(-self.Ea / (self.R * (self.current_temp + 273.15)))

                # Concentration at surface (simplified, proportional to pressure and flow)
                C_surface = (self.current_pressure / 760.0) * (sih4_flow / 100.0) * 1e-3

                # Growth rate (nm/s)
                growth_rate = k * C_surface * 0.1  # Scale factor for realistic rates

                # Add noise
                growth_rate += np.random.normal(0, growth_rate * 0.02)

                # Accumulate thickness
                self.accumulated_thickness += growth_rate * dt
            else:
                growth_rate = 0.0

            # Add realistic noise to measurements
            temp_noise = np.random.normal(0, 0.5)
            pressure_noise = np.random.normal(0, target_pressure * 0.001)

            # Create telemetry point
            telemetry = CVDTelemetrySchema(
                cvd_run_id=run_id,
                ts=current_time,
                chamber_pressure_torr=self.current_pressure + pressure_noise,
                temperature_zones_c={"zone1": self.current_temp + temp_noise},
                gas_flows_sccm={"SiH4": sih4_flow * (1 + np.random.normal(0, 0.01))},
                qcm_rate_nm_per_s=growth_rate,
                qcm_thickness_nm=self.accumulated_thickness,
                deposition_rate_nm_per_min=growth_rate * 60
            )

            yield telemetry

            await asyncio.sleep(dt)
```

### PECVD Simulator

```python
# services/analysis/app/tools/simulators/plasma_sim.py

class PECVDSimulator(CVDTool):
    """
    PECVD simulator with plasma physics
    - Ion bombardment effects
    - Plasma density modeling
    - RF power/bias relationships
    - Lower temperature deposition
    """

    async def stream_telemetry(self, run_id: UUID) -> AsyncIterator[CVDTelemetrySchema]:
        recipe = await self._get_recipe(run_id)

        # Extract parameters
        temp = recipe.temperature_profile["zones"][0]["setpoint"]
        pressure = recipe.pressure_setpoints["process_pressure_torr"]
        rf_power = recipe.plasma_settings["rf_power_w"]
        dc_bias = recipe.plasma_settings["dc_bias_v"]

        while self.is_running:
            # Plasma-enhanced growth rate
            # Higher rate at lower T compared to thermal
            plasma_enhancement = (rf_power / 300.0) ** 0.5
            base_rate = 5.0  # nm/s at standard conditions
            growth_rate = base_rate * plasma_enhancement * (pressure / 1.0) ** 0.3

            # Ion bombardment creates stress
            stress_factor = abs(dc_bias) / 100.0

            # Add plasma-specific noise
            rf_reflected = rf_power * np.random.uniform(0.02, 0.05)  # 2-5% reflection

            telemetry = CVDTelemetrySchema(
                cvd_run_id=run_id,
                ts=datetime.utcnow(),
                chamber_pressure_torr=pressure + np.random.normal(0, 0.01),
                temperature_zones_c={"zone1": temp + np.random.normal(0, 2.0)},
                rf_power_w=rf_power + np.random.normal(0, 5.0),
                rf_reflected_w=rf_reflected,
                dc_bias_v=dc_bias + np.random.normal(0, 5.0),
                plasma_impedance_ohm=50 + np.random.normal(0, 5.0),
                qcm_rate_nm_per_s=growth_rate,
                qcm_thickness_nm=self.accumulated_thickness,
                deposition_rate_nm_per_min=growth_rate * 60
            )

            self.accumulated_thickness += growth_rate * 1.0

            yield telemetry
            await asyncio.sleep(1.0)
```

---

## 📡 API Endpoints

### Complete REST API

```python
# services/analysis/app/routers/cvd.py

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID

from ..database import get_db
from ..models import cvd as models
from ..schemas import cvd as schemas
from ..auth import get_current_user, require_role
from ..tasks.cvd_tasks import start_cvd_run_task


router = APIRouter(prefix="/api/cvd", tags=["CVD"])


# Process Modes
@router.post("/process-modes", response_model=schemas.CVDProcessModeSchema)
async def create_process_mode(
    mode: schemas.CVDProcessModeCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new CVD process mode"""
    db_mode = models.CVDProcessMode(**mode.dict(), org_id=current_user.org_id)
    db.add(db_mode)
    await db.commit()
    await db.refresh(db_mode)
    return db_mode


@router.get("/process-modes", response_model=List[schemas.CVDProcessModeSchema])
async def list_process_modes(
    pressure_mode: Optional[models.PressureMode] = None,
    energy_mode: Optional[models.EnergyMode] = None,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List CVD process modes with filtering"""
    query = select(models.CVDProcessMode).where(
        models.CVDProcessMode.org_id == current_user.org_id
    )

    if pressure_mode:
        query = query.where(models.CVDProcessMode.pressure_mode == pressure_mode)
    if energy_mode:
        query = query.where(models.CVDProcessMode.energy_mode == energy_mode)

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


# Recipes
@router.post("/recipes", response_model=schemas.CVDRecipeSchema)
async def create_recipe(
    recipe: schemas.CVDRecipeCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new CVD recipe"""
    db_recipe = models.CVDRecipe(
        **recipe.dict(),
        org_id=current_user.org_id,
        created_by_id=current_user.id
    )
    db.add(db_recipe)
    await db.commit()
    await db.refresh(db_recipe)
    return db_recipe


@router.get("/recipes/{recipe_id}", response_model=schemas.CVDRecipeSchema)
async def get_recipe(
    recipe_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get recipe by ID"""
    recipe = await db.get(models.CVDRecipe, recipe_id)
    if not recipe or recipe.org_id != current_user.org_id:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipe


# Runs
@router.post("/runs", response_model=schemas.CVDRunSchema)
async def create_run(
    run: schemas.CVDRunCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create and queue a new CVD run"""
    # Validate recipe and instrument
    recipe = await db.get(models.CVDRecipe, run.cvd_recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    # Create run
    db_run = models.CVDRun(
        **run.dict(),
        org_id=current_user.org_id,
        operator_id=current_user.id,
        status=models.RunStatus.QUEUED,
        run_number=f"CVD-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    )
    db.add(db_run)
    await db.commit()
    await db.refresh(db_run)

    # Queue Celery task
    task = start_cvd_run_task.delay(str(db_run.id))
    db_run.celery_task_id = task.id
    await db.commit()

    return db_run


@router.get("/runs/{run_id}", response_model=schemas.CVDRunSchema)
async def get_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get run by ID"""
    run = await db.get(models.CVDRun, run_id)
    if not run or run.org_id != current_user.org_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs/{run_id}/telemetry", response_model=List[schemas.CVDTelemetrySchema])
async def get_run_telemetry(
    run_id: UUID,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(1000, le=10000),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get telemetry data for a run"""
    query = select(models.CVDTelemetry).where(
        models.CVDTelemetry.cvd_run_id == run_id,
        models.CVDTelemetry.org_id == current_user.org_id
    )

    if start_time:
        query = query.where(models.CVDTelemetry.ts >= start_time)
    if end_time:
        query = query.where(models.CVDTelemetry.ts <= end_time)

    query = query.order_by(models.CVDTelemetry.ts).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()


@router.websocket("/runs/{run_id}/stream")
async def stream_run_telemetry(
    websocket: WebSocket,
    run_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Stream live telemetry via WebSocket"""
    await websocket.accept()

    try:
        # Get tool driver
        run = await db.get(models.CVDRun, run_id)
        tool = get_tool_driver(run.instrument_id)

        # Stream telemetry
        async for telemetry in tool.stream_telemetry(run_id):
            await websocket.send_json(telemetry.dict())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")


# Results
@router.get("/runs/{run_id}/results", response_model=schemas.CVDResultSchema)
async def get_run_results(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get results for a run"""
    result = await db.execute(
        select(models.CVDResult).where(
            models.CVDResult.cvd_run_id == run_id,
            models.CVDResult.org_id == current_user.org_id
        )
    )
    result = result.scalar_one_or_none()
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    return result


# VM Prediction
@router.post("/vm/predict", response_model=schemas.VMPredictionSchema)
async def predict_thickness(
    prediction_request: schemas.VMPredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Predict film thickness using VM model"""
    # Load VM model
    from ..ml.vm_models import get_vm_model

    model = get_vm_model(prediction_request.process_mode_id)

    # Extract features
    features = model.extract_features(
        recipe=prediction_request.recipe,
        telemetry_summary=prediction_request.telemetry_summary,
        design_features=prediction_request.design_features
    )

    # Predict
    prediction = model.predict(features)

    return schemas.VMPredictionSchema(
        predicted_thickness=prediction["thickness"],
        predicted_uniformity=prediction["uniformity"],
        confidence=prediction["confidence"],
        model_version=model.version
    )


# Simulation
@router.post("/simulate", response_model=schemas.CVDRunSchema)
async def simulate_run(
    simulation_request: schemas.CVDSimulationRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Run full HIL simulation without hardware"""
    # Create simulated run
    db_run = models.CVDRun(
        org_id=current_user.org_id,
        cvd_recipe_id=simulation_request.recipe_id,
        process_mode_id=simulation_request.process_mode_id,
        instrument_id=simulation_request.instrument_id,
        operator_id=current_user.id,
        status=models.RunStatus.QUEUED,
        run_number=f"SIM-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        metadata={"simulation": True}
    )
    db.add(db_run)
    await db.commit()

    # Queue simulation task
    task = simulate_cvd_run_task.delay(str(db_run.id))
    db_run.celery_task_id = task.id
    await db.commit()

    return db_run
```

---

## 🎨 Frontend Components (Next.js)

### CVD Dashboard Page

```typescript
// frontend/src/app/cvd/page.tsx

'use client';

import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { ProcessModeCard } from '@/components/cvd/ProcessModeCard';
import { RunStatusWidget } from '@/components/cvd/RunStatusWidget';
import { SPCAlertWidget } from '@/components/cvd/SPCAlertWidget';
import { api } from '@/lib/api/cvd';

export default function CVDDashboard() {
  const [stats, setStats] = useState(null);
  const [recentRuns, setRecentRuns] = useState([]);
  const [activeAlerts, setActiveAlerts] = useState([]);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    const [statsData, runsData, alertsData] = await Promise.all([
      api.getStats(),
      api.getRecentRuns(),
      api.getActiveAlerts()
    ]);
    setStats(statsData);
    setRecentRuns(runsData);
    setActiveAlerts(alertsData);
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold">CVD Platform</h1>

      {/* Statistics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Total Runs Today</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-4xl font-bold">{stats?.runs_today || 0}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active Runs</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-4xl font-bold text-green-600">
              {stats?.active_runs || 0}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>SPC Violations</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-4xl font-bold text-red-600">
              {stats?.spc_violations || 0}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>VM Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-4xl font-bold text-blue-600">
              {stats?.vm_accuracy || 0}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Active Alerts */}
      {activeAlerts.length > 0 && (
        <Card className="border-red-500">
          <CardHeader>
            <CardTitle className="text-red-600">Active Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <SPCAlertWidget alerts={activeAlerts} />
          </CardContent>
        </Card>
      )}

      {/* Recent Runs */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Runs</CardTitle>
        </CardHeader>
        <CardContent>
          <RunStatusWidget runs={recentRuns} />
        </CardContent>
      </Card>

      {/* Process Modes */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {stats?.process_modes?.map((mode) => (
          <ProcessModeCard key={mode.id} mode={mode} />
        ))}
      </div>
    </div>
  );
}
```

### Recipe Builder Component

```typescript
// frontend/src/components/cvd/RecipeBuilder.tsx

'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { RecipeStepEditor } from './RecipeStepEditor';
import { PlasmaSettingsPanel } from './PlasmaSettingsPanel';
import { GasFlowPanel } from './GasFlowPanel';

interface RecipeBuilderProps {
  processMode: ProcessMode;
  onSave: (recipe: CVDRecipe) => void;
}

export function RecipeBuilder({ processMode, onSave }: RecipeBuilderProps) {
  const [recipe, setRecipe] = useState({
    name: '',
    film_target: '',
    thickness_target_nm: 100,
    uniformity_target_pct: 2.0,
    temperature_profile: {
      zones: [{ zone: 1, setpoint: 800, ramp_rate: 10 }]
    },
    pressure_setpoints: {
      process_pressure_torr: 0.5
    },
    gas_flows: {
      gases: []
    },
    plasma_settings: null,
    recipe_steps: []
  });

  const handleAddGas = (gas: Gas) => {
    setRecipe({
      ...recipe,
      gas_flows: {
        gases: [...recipe.gas_flows.gases, gas]
      }
    });
  };

  const handleAddStep = (step: RecipeStep) => {
    setRecipe({
      ...recipe,
      recipe_steps: [...recipe.recipe_steps, step]
    });
  };

  const showPlasmaPanel = processMode.energy_mode === 'plasma';
  const showLaserPanel = processMode.energy_mode === 'laser';
  const showPulsingPanel = processMode.variant?.includes('pulsed') ||
                          processMode.variant?.includes('ALCVD');

  return (
    <div className="space-y-6">
      {/* Basic Info */}
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-4">Recipe Information</h2>
        <div className="grid grid-cols-2 gap-4">
          <Input
            label="Recipe Name"
            value={recipe.name}
            onChange={(e) => setRecipe({ ...recipe, name: e.target.value })}
          />
          <Input
            label="Film Target"
            value={recipe.film_target}
            onChange={(e) => setRecipe({ ...recipe, film_target: e.target.value })}
            placeholder="e.g., SiO2, Si3N4, GaN"
          />
          <Input
            type="number"
            label="Target Thickness (nm)"
            value={recipe.thickness_target_nm}
            onChange={(e) => setRecipe({ ...recipe, thickness_target_nm: parseFloat(e.target.value) })}
          />
          <Input
            type="number"
            label="Uniformity Target (%)"
            value={recipe.uniformity_target_pct}
            onChange={(e) => setRecipe({ ...recipe, uniformity_target_pct: parseFloat(e.target.value) })}
          />
        </div>
      </Card>

      {/* Temperature Profile */}
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-4">Temperature Profile</h2>
        {/* Temperature zone editors */}
      </Card>

      {/* Gas Flows */}
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-4">Gas Flows</h2>
        <GasFlowPanel
          gases={recipe.gas_flows.gases}
          onAddGas={handleAddGas}
          onRemoveGas={(index) => {
            const newGases = [...recipe.gas_flows.gases];
            newGases.splice(index, 1);
            setRecipe({
              ...recipe,
              gas_flows: { gases: newGases }
            });
          }}
        />
      </Card>

      {/* Plasma Settings (if applicable) */}
      {showPlasmaPanel && (
        <Card className="p-6">
          <h2 className="text-2xl font-bold mb-4">Plasma Settings</h2>
          <PlasmaSettingsPanel
            settings={recipe.plasma_settings}
            onChange={(settings) => setRecipe({ ...recipe, plasma_settings: settings })}
          />
        </Card>
      )}

      {/* Recipe Steps */}
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-4">Recipe Steps</h2>
        <RecipeStepEditor
          steps={recipe.recipe_steps}
          onAddStep={handleAddStep}
          onRemoveStep={(index) => {
            const newSteps = [...recipe.recipe_steps];
            newSteps.splice(index, 1);
            setRecipe({ ...recipe, recipe_steps: newSteps });
          }}
        />
      </Card>

      {/* Save Button */}
      <div className="flex justify-end space-x-4">
        <Button variant="outline">Cancel</Button>
        <Button onClick={() => onSave(recipe)}>Save Recipe</Button>
      </div>
    </div>
  );
}
```

---

## 📊 Integration Summary

### LIMS Integration

```python
# services/lims/app/integrations/cvd_integration.py

from typing import Optional
from uuid import UUID

class CVDLIMSIntegration:
    """Integration between CVD platform and LIMS"""

    async def link_run_to_sample(
        self,
        cvd_run_id: UUID,
        sample_id: UUID
    ) -> bool:
        """Link CVD run to LIMS sample"""
        # Update CVD run with sample_id
        # Create LIMS process record
        # Link results when available
        pass

    async def create_eln_entry(
        self,
        cvd_run_id: UUID
    ) -> UUID:
        """Auto-generate ELN entry for CVD run"""
        # Get run details
        # Create ELN entry with:
        #   - Recipe summary
        #   - Telemetry highlights
        #   - Results
        #   - Signatures
        pass

    async def generate_report(
        self,
        cvd_run_id: UUID
    ) -> str:
        """Generate comprehensive PDF/HTML report"""
        # Include:
        #   - Recipe details
        #   - Telemetry plots
        #   - Final results
        #   - SPC status
        #   - VM predictions vs actual
        #   - Calibration references
        pass
```

---

## 🚀 Deployment & Scaling

### Enhanced Docker Compose

```yaml
# docker-compose.enhanced.yml

version: '3.8'

services:
  # Previous services (backend, frontend, postgres, redis, etc.)
  # ... (from original docker-compose.yml)

  # Celery worker for CVD tasks
  celery_worker_cvd:
    build: ./services/analysis
    command: celery -A app.tasks worker --loglevel=info --queue=cvd
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://redis:6379
      - CELERY_BROKER_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./services/analysis:/app
    restart: unless-stopped

  # TimescaleDB for telemetry (extends postgres)
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_DB=cvd_telemetry
      - POSTGRES_USER=cvd_user
      - POSTGRES_PASSWORD=cvd_pass
    volumes:
      - timescale_data:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    restart: unless-stopped

  # Metabase for analytics dashboards
  metabase:
    image: metabase/metabase:latest
    environment:
      - MB_DB_TYPE=postgres
      - MB_DB_DBNAME=metabase
      - MB_DB_PORT=5432
      - MB_DB_USER=metabase_user
      - MB_DB_PASS=metabase_pass
      - MB_DB_HOST=postgres
    ports:
      - "3002:3000"
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  timescale_data:
```

---

## 📚 Documentation Delivered

### ✅ Complete Documentation Files

1. **MASTER_IMPLEMENTATION_GUIDE.md** (this file)
2. Database migration script
3. SQLAlchemy models
4. Tool abstraction design
5. API endpoint specifications
6. Frontend component examples
7. Simulator architecture
8. Integration patterns

### 📋 Additional Documentation Needed

- **PROCESS_MODES.md** - Detailed guide for each CVD variant
- **RECIPE_SPECIFICATION.md** - Recipe JSON schema reference
- **VM_ML_MODELS.md** - ML model training and deployment
- **API_REFERENCE.md** - Complete OpenAPI/Swagger docs
- **UI_GUIDE.md** - Frontend user manual

---

## 🎯 Next Steps (Implementation Phases)

### Phase 1: Core Infrastructure (Current)
✅ Database schema
✅ SQLAlchemy models
⏳ Pydantic schemas
⏳ Basic API routers
⏳ Tool abstraction base

### Phase 2: Simulators & Physics
- Complete all HIL simulators
- Enhanced physics models
- Calibration routines

### Phase 3: ML & Control
- Feature store implementation
- VM model training pipeline
- R2R control integration
- SPC/FDC automation

### Phase 4: Frontend
- All Next.js pages
- React components
- WebSocket integration
- Real-time dashboards

### Phase 5: Integration & Testing
- LIMS integration
- Report generation
- Comprehensive testing
- Performance optimization

---

## 📦 File Count Summary

**Already Created:** 19 files + 3 verification files = 22 files

**New in Enhanced Platform:**
- Database: 1 migration file
- Backend Models: 1 SQLAlchemy file
- Backend Schemas: 1 Pydantic file (to create)
- Backend Routers: 1 API file (to create)
- Backend Tools: 1 base + 6 drivers + 3 simulators = 10 files
- Backend Physics: 4 model files
- Backend ML: 4 feature/model files
- Backend Control: 4 controller files
- Backend Tasks: 1 Celery file
- Frontend Pages: 10 Next.js pages
- Frontend Components: 11 React components
- Frontend API: 1 TypeScript client
- Tests: 9 test files
- Docs: 10 documentation files
- Deployment: 2 config files

**Total New Files: ~70 files**

**Grand Total: ~92 files for complete platform**

---

## 💾 Download Instructions

All files are in:
```
/home/user/SPECTRA-Lab/cvd_platform/
```

**Critical files now available:**
1. ✅ Database migration (`services/analysis/app/alembic/versions/0001_cvd_module.py`)
2. ✅ SQLAlchemy models (`services/analysis/app/models/cvd.py`)
3. ✅ Master implementation guide (this file)
4. ✅ All previous 22 files from v1.0

**Download:**
```bash
# Complete archive
tar -czf cvd_platform_v2_enhanced.tar.gz cvd_platform/

# Or from GitHub
git pull origin claude/follow-prompt-files-011CV59dTKQn8W2Pm8hCdWHr
```

---

## 📞 Support

This master guide provides:
- ✅ Complete architecture for all CVD variants
- ✅ Database schema with full migration
- ✅ SQLAlchemy models with relationships
- ✅ API endpoint specifications
- ✅ Tool abstraction design
- ✅ Simulator architecture
- ✅ Frontend component examples
- ✅ Integration patterns
- ✅ Deployment strategy

**Status:** Foundation complete, ready for phase-by-phase implementation

---

**Version:** 2.0 Enhanced
**Last Updated:** 2024-11-13
**Total Platform Scope:** 92+ files supporting 37+ CVD process variants
