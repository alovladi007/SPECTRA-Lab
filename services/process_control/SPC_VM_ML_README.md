# SPC & VM/ML Integration Documentation

Complete implementation of Statistical Process Control (SPC) and Virtual Metrology (VM) with MLOps infrastructure for Ion Implantation and Rapid Thermal Processing (RTP) systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [SPC Module](#spc-module)
4. [Virtual Metrology Module](#virtual-metrology-module)
5. [MLOps Infrastructure](#mlops-infrastructure)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)
8. [Deployment Guide](#deployment-guide)

---

## Overview

This implementation provides production-ready SPC monitoring, virtual metrology prediction, and MLOps infrastructure for semiconductor manufacturing process control.

### Key Features

**Statistical Process Control (SPC)**
- X-bar/R charts for process mean and range monitoring
- EWMA charts for detecting small process shifts
- CUSUM charts for sustained drift detection
- Western Electric rules (8 rules) for out-of-control detection
- Alert deduplication and severity classification
- Root Cause Analysis (RCA) playbooks with corrective actions

**Virtual Metrology (VM)**
- Ion Implant VM: predict sheet resistance, junction depth, activation
- RTP VM: predict dopant activation, diffusion, oxide thickness
- Physics-based feature engineering
- Uncertainty quantification
- Feature importance analysis

**MLOps Infrastructure**
- Feature store with online/offline serving
- Model registry with versioning and staging
- Drift monitoring (feature, prediction, concept drift)
- Multiple deployment strategies (direct, shadow, canary, blue-green)
- Model cards for responsible AI documentation
- ONNX export for cross-platform deployment

---

## Architecture

```
services/process_control/
├── app/
│   ├── spc/                    # Statistical Process Control
│   │   ├── charts.py           # X-bar/R, EWMA, CUSUM charts
│   │   ├── monitors.py         # Ion & RTP process monitors
│   │   ├── rca.py              # Root cause analysis
│   │   └── __init__.py
│   │
│   ├── vm/                     # Virtual Metrology
│   │   ├── ion_vm.py           # Ion Implant VM model
│   │   ├── rtp_vm.py           # RTP VM model
│   │   └── __init__.py
│   │
│   └── mlops/                  # MLOps Infrastructure
│       ├── feature_store.py    # Feature management
│       ├── model_registry.py   # Model lifecycle
│       ├── drift_monitor.py    # Drift detection
│       ├── onnx_export.py      # ONNX conversion
│       └── __init__.py
│
└── tests/
    └── unit/
        ├── spc/                # SPC unit tests
        ├── vm/                 # VM unit tests
        └── mlops/              # MLOps unit tests
```

---

## SPC Module

### Statistical Charts

#### X-bar/R Chart

Monitors process mean (X-bar) and range (R) for detecting process shifts and variability changes.

**Theory:**
- X-bar chart: UCL = μ + A₂·R̄, LCL = μ - A₂·R̄
- R chart: UCL = D₄·R̄, LCL = D₃·R̄

**Usage:**
```python
from app.spc import XbarRChart

chart = XbarRChart(
    parameter_name="beam_current_mA",
    subgroup_size=5,      # Samples per subgroup
    window_size=25        # Rolling window for control limits
)

# Update with new subgroup
chart.update([0.98, 1.01, 0.99, 1.02, 1.00], timestamp=time.time())

# Check for alerts
alerts = chart.get_alerts()
```

#### EWMA Chart

Exponentially Weighted Moving Average for detecting small process shifts.

**Theory:**
- EWMA: z_t = λ·x_t + (1-λ)·z_{t-1}
- Control limits: UCL/LCL = μ ± L·σ·√(λ/(2-λ)·[1-(1-λ)^(2t)])

**Usage:**
```python
from app.spc import EWMAChart

chart = EWMAChart(
    parameter_name="dose_uniformity_pct",
    lambda_weight=0.2,    # Smoothing factor (0-1)
    L=3.0                 # Control limit multiplier
)

chart.update(95.2, timestamp=time.time())
```

#### CUSUM Chart

Cumulative Sum chart for detecting sustained process shifts.

**Theory:**
- CUSUM High: C⁺_t = max(0, C⁺_{t-1} + x_t - (μ + K))
- CUSUM Low: C⁻_t = max(0, C⁻_{t-1} - x_t + (μ - K))
- Alert when C⁺ or C⁻ exceeds decision interval H

**Usage:**
```python
from app.spc import CUSUMChart

chart = CUSUMChart(
    parameter_name="source_pressure_mtorr",
    k=0.5,               # Slack parameter (half-sigma shifts)
    h=5.0                # Decision interval
)

chart.update(2.1e-5, timestamp=time.time())
```

### Western Electric Rules

Eight rules for detecting out-of-control conditions:

1. **Rule 1**: One point beyond 3σ from center line
2. **Rule 2**: Nine consecutive points on same side of center line
3. **Rule 3**: Six consecutive points all increasing or decreasing
4. **Rule 4**: Fourteen consecutive points alternating up/down
5. **Rule 5**: Two of three consecutive points beyond 2σ (same side)
6. **Rule 6**: Four of five consecutive points beyond 1σ (same side)
7. **Rule 7**: Fifteen consecutive points within 1σ (both sides)
8. **Rule 8**: Eight consecutive points beyond 1σ (both sides)

### Process Monitors

#### Ion Implant Monitor

Monitors key Ion Implantation parameters with preconfigured defaults:

```python
from app.spc import IonImplantMonitor, IonParameter

monitor = IonImplantMonitor()

# Start monitoring a recipe
monitor.start_recipe(recipe_id="ION_RECIPE_001")

# Update with measurements
measurements = {
    IonParameter.BEAM_CURRENT_MA: 1.02,
    IonParameter.DOSE_UNIFORMITY_PCT: 94.8,
    IonParameter.SOURCE_PRESSURE_MTORR: 2.3e-5,
    IonParameter.ANALYZER_FIELD_GAUSS: 1520,
}

alerts = monitor.update(measurements, timestamp=time.time())

# Get summary
summary = monitor.get_summary()
print(f"Total alerts: {summary['total_alerts']}")
```

**Default Parameters:**
- `BEAM_CURRENT_MA`: EWMA, λ=0.2, critical threshold ±15%
- `DOSE_UNIFORMITY_PCT`: X-bar/R, target=95%, LCL=90%
- `SOURCE_PRESSURE_MTORR`: CUSUM, k=0.5, h=5.0
- `ANALYZER_FIELD_GAUSS`: X-bar/R, subgroup size=5

#### RTP Monitor

Monitors Rapid Thermal Processing parameters:

```python
from app.spc import RTPMonitor, RTPParameter

monitor = RTPMonitor()

monitor.start_recipe(recipe_id="RTP_RECIPE_001")

measurements = {
    RTPParameter.RAMP_TRACKING_ERROR_C: 3.2,
    RTPParameter.OVERSHOOT_PCT: 1.8,
    RTPParameter.LAMP_POWER_PCT: 87.5,
    RTPParameter.EMISSIVITY: 0.68,
    RTPParameter.GAS_FLOW_DEVIATION_SCCM: 2.1,
}

alerts = monitor.update(measurements, timestamp=time.time())
```

**Default Parameters:**
- `RAMP_TRACKING_ERROR_C`: EWMA, λ=0.2, critical threshold ±10°C
- `OVERSHOOT_PCT`: X-bar/R, target=0%, UCL=5%
- `LAMP_POWER_PCT`: CUSUM, k=0.5, h=5.0
- `EMISSIVITY`: EWMA, λ=0.15, range=[0.6, 0.8]
- `GAS_FLOW_DEVIATION_SCCM`: X-bar/R, target=0, UCL=5.0

### Root Cause Analysis

Automated RCA with playbooks for common failure modes:

```python
from app.spc import AlertTriageEngine

triage_engine = AlertTriageEngine()

# Triage an alert
result = triage_engine.triage_alert(alert)

print(f"Likely causes: {result.likely_causes}")
print(f"Severity: {result.severity}")
print(f"Recommended actions:")
for action in result.recommended_actions[:3]:
    print(f"  - {action.description} (priority {action.priority})")
```

**Ion Implant RCA Playbooks:**
1. Beam current instability → Source degradation, optics misalignment, vacuum leak
2. Dose uniformity issues → Scan system malfunction, wafer handling, beam profile
3. Vacuum degradation → Pump failure, leak, outgassing
4. Energy drift → HV supply drift, analyzer misalignment

**RTP RCA Playbooks:**
1. Ramp tracking error → Pyrometer drift, lamp degradation, thermal mass change
2. Overshoot → PID tuning, lamp response, emissivity mismatch
3. Lamp power anomaly → Lamp aging, reflector degradation, power supply
4. Emissivity drift → Surface contamination, backside coating, wafer type change
5. Gas flow deviation → MFC failure, pressure control, valve malfunction

---

## Virtual Metrology Module

### Ion Implant VM

Predicts post-implant electrical properties without physical metrology.

**Inputs:**
- Ion species (B, P, As, Sb, In, etc.)
- Energy (keV)
- Dose (ions/cm²)
- Tilt/twist angles
- RTP thermal budget (if applicable)

**Outputs:**
- Sheet resistance (Ω/□)
- Junction depth (μm)
- Activation percentage (%)
- Quality score (0-100)
- Uncertainty estimates
- Feature importance

**Usage:**
```python
from app.vm import IonVirtualMetrologyModel, IonVMFeatures

model = IonVirtualMetrologyModel()

features = IonVMFeatures(
    ion_species="P",          # Phosphorus
    energy_keV=30.0,
    dose_cm2=5e15,
    tilt_angle_deg=7.0,
    twist_angle_deg=0.0,
    # RTP thermal info (optional)
    rtp_peak_temp_C=1000.0,
    rtp_dwell_time_s=30.0,
    rtp_thermal_budget=1.2e6
)

prediction = model.predict(features)

print(f"Sheet Resistance: {prediction.sheet_resistance_ohm_sq:.2f} Ω/□")
print(f"Junction Depth: {prediction.junction_depth_um:.3f} μm")
print(f"Activation: {prediction.activation_pct:.1f}%")
print(f"Quality Score: {prediction.quality_score}")
```

**Physics-Based Features:**
- LSS theory for range estimation
- Caughey-Thomas mobility model
- Arrhenius activation kinetics
- Dose-energy interaction terms
- Angular dependence (channeling)

### RTP VM

Predicts thermal processing outcomes for dopant activation and diffusion.

**Inputs:**
- Peak temperature (°C)
- Dwell time (s)
- Ramp rate (°C/s)
- Ambient gas (N₂, O₂, Ar)
- Process response (tracking error, overshoot)

**Outputs:**
- Activation percentage (%)
- Diffusion depth (μm)
- Sheet resistance change (%)
- Oxide thickness (Å, for O₂ ambient)
- Quality score (0-100)

**Usage:**
```python
from app.vm import RTPVirtualMetrologyModel, RTPVMFeatures

model = RTPVirtualMetrologyModel()

features = RTPVMFeatures(
    peak_temp_C=1000.0,
    dwell_time_s=30.0,
    ramp_rate_C_per_s=50.0,
    ambient_gas="N2",
    gas_flow_sccm=5000.0,
    # Process response
    avg_tracking_error_C=2.5,
    max_overshoot_C=8.0,
    thermal_budget=1.25e6
)

prediction = model.predict(features)

print(f"Activation: {prediction.activation_pct:.1f}%")
print(f"Diffusion Depth: {prediction.diffusion_depth_um:.3f} μm")
print(f"Rs Change: {prediction.rs_change_pct:+.1f}%")
```

**Physics Models:**
- Fick's 2nd law for diffusion
- Arrhenius temperature dependence
- Deal-Grove oxide growth model
- Dopant-specific parameters (D₀, Eₐ)

---

## MLOps Infrastructure

### Feature Store

Manages features for online (real-time inference) and offline (batch training) serving.

**Architecture:**
- **Online Store**: Latest features for real-time predictions (JSON files, Redis in production)
- **Offline Store**: Historical features for training (Pickle, Parquet in production)
- **Definitions**: Feature schemas with validation rules

**Usage:**

```python
from app.mlops import FeatureStore, create_ion_implant_feature_group

# Initialize
feature_store = FeatureStore(store_dir="./feature_store")

# Register feature group
ion_features = create_ion_implant_feature_group()
feature_store.register_feature_group(ion_features)

# Write features for online serving
features = {
    "ion_species_code": 5,  # Phosphorus
    "energy_keV": 30.0,
    "dose_cm2": 5e15,
    "tilt_angle_deg": 7.0,
    "beam_current_mA": 1.02,
    "dose_uniformity_pct": 95.2,
    "rtp_thermal_budget": 1.2e6
}

feature_store.write_online(
    entity_id="RUN_12345_WAFER_01",
    features=features,
    feature_group="ion_implant_features"
)

# Read features for inference
serving_features = feature_store.read_online(
    entity_id="RUN_12345_WAFER_01",
    feature_group="ion_implant_features"
)

# Write to offline store for training
from app.mlops import FeatureVector
vectors = [
    FeatureVector(
        entity_id="RUN_12345_WAFER_01",
        timestamp=time.time(),
        features=features,
        feature_group="ion_implant_features"
    ),
    # ... more vectors
]

feature_store.write_offline(vectors, "ion_implant_features")

# Get training dataset
dataset = feature_store.get_training_dataset(
    feature_groups=["ion_implant_features"],
    start_date="2025-01-01",
    end_date="2025-02-01"
)
```

**Predefined Feature Groups:**
- `ion_implant_features`: 7 features (species, energy, dose, tilt, beam current, uniformity, thermal budget)
- `rtp_features`: 6 features (peak temp, dwell time, ramp rate, thermal budget, tracking error, overshoot)

### Model Registry

Manages ML model lifecycle with versioning, staging, and deployment.

**Lifecycle Stages:**
- `DEVELOPMENT`: Model under development
- `STAGING`: Model in staging for validation
- `PRODUCTION`: Live production model
- `ARCHIVED`: Deprecated model

**Deployment Strategies:**
- `DIRECT`: Replace immediately
- `SHADOW`: Run in parallel, log predictions only
- `CANARY`: Route small % of traffic (e.g., 5%)
- `BLUE_GREEN`: Switch between two versions

**Usage:**

```python
from app.mlops import ModelRegistry, ModelMetadata, ModelCard, ModelStage, DeploymentStrategy

# Initialize
registry = ModelRegistry(registry_dir="./model_registry")

# Register a new model
metadata = ModelMetadata(
    model_name="ion_vm",
    model_version="v1.2.0",
    model_type="ion_vm",
    stage=ModelStage.DEVELOPMENT,
    training_date="2025-01-15",
    training_samples=5000,
    training_r2_score=0.92,
    validation_r2_score=0.89,
    model_file_path="./models/ion_vm_v1.2.0.pkl",
    model_file_hash="",  # Calculated automatically
    mae=12.5,
    rmse=18.3,
    description="Random forest with 100 estimators",
    tags=["ion", "vm", "random_forest"]
)

# Create model card
card = ModelCard(
    model_name="ion_vm",
    model_version="v1.2.0",
    model_type="ion_vm",
    intended_use="Predict sheet resistance for Ion Implant processes",
    training_algorithm="random_forest",
    training_data_description="5000 wafers from Fab A, tools ION-01 to ION-04",
    training_data_size=5000,
    training_data_date_range="2024-06-01 to 2025-01-01",
    feature_names=["ion_species_code", "energy_keV", "dose_cm2", ...],
    performance_metrics={
        "r2_score": 0.89,
        "mae": 12.5,
        "rmse": 18.3
    },
    known_limitations=[
        "Performance degrades for energies >100 keV",
        "Limited data for Indium implants"
    ],
    responsible_party="ML Engineering Team",
    contact="ml-team@fab.com"
)

key = registry.register_model(
    model_file_path="./models/ion_vm_v1.2.0.pkl",
    metadata=metadata,
    model_card=card
)

# Promote to staging
registry.promote_model("ion_vm", "v1.2.0", ModelStage.STAGING)

# Deploy to production with canary strategy
registry.deploy_model(
    model_name="ion_vm",
    version="v1.2.0",
    strategy=DeploymentStrategy.CANARY,
    deployed_by="john.doe@fab.com"
)

# Compare models
comparison = registry.compare_models(
    "ion_vm", "v1.1.0",
    "ion_vm", "v1.2.0"
)
print(f"Winner by R²: {comparison['winner_by_r2']}")
```

### Drift Monitoring

Detects feature drift, prediction drift, and concept drift using statistical tests.

**Detection Methods:**
- **Kolmogorov-Smirnov Test**: Distribution comparison (p-value < 0.05 = drift)
- **Population Stability Index (PSI)**: Binned distribution shift
  - PSI < 0.1: No drift
  - 0.1 ≤ PSI < 0.25: Moderate drift
  - PSI ≥ 0.25: Significant drift
- **Jensen-Shannon Divergence**: Symmetric KL divergence

**Usage:**

```python
from app.mlops import ModelDriftMonitor

# Initialize monitor
monitor = ModelDriftMonitor(
    model_name="ion_vm",
    feature_names=["energy_keV", "dose_cm2", "tilt_angle_deg", ...],
    baseline_window_size=1000,
    current_window_size=100
)

# Update with new prediction
features = np.array([30.0, 5e15, 7.0, ...])
prediction = 150.5  # Predicted sheet resistance
ground_truth = 148.2  # Actual measurement (if available)

alerts = monitor.update(features, prediction, ground_truth)

for alert in alerts:
    print(f"DRIFT ALERT: {alert.drift_type}")
    print(f"  Feature: {alert.feature_name}")
    print(f"  Severity: {alert.severity}")
    print(f"  PSI: {alert.psi_score:.3f}")
    print(f"  Recommended: {alert.recommended_action}")

# Get drift summary
summary = monitor.get_drift_summary()
print(f"Total alerts: {len(summary['all_alerts'])}")
print(f"Feature drift count: {summary['drift_counts']['feature']}")
```

**Alert Severity:**
- `LOW`: PSI 0.25-0.35, informational
- `MEDIUM`: PSI 0.35-0.5, investigate
- `HIGH`: PSI 0.5-0.75, retrain model
- `CRITICAL`: PSI ≥ 0.75, immediate action

### ONNX Export

Export trained models to ONNX format for cross-platform deployment.

**Supported Frameworks:**
- scikit-learn (RandomForest, GradientBoosting, etc.)
- XGBoost
- PyTorch

**Usage:**

```python
from app.mlops import ONNXExporter, ONNXExportConfig, ONNXInferenceRuntime
import numpy as np

# Initialize exporter
exporter = ONNXExporter(export_dir="./onnx_models")

# Export scikit-learn model
from sklearn.ensemble import RandomForestRegressor

# Assume `model` is trained RandomForest
config = ONNXExportConfig(
    target_opset=14,
    optimize=True,
    validate=True,
    test_inputs=X_test  # For validation
)

metadata = exporter.export_sklearn_model(
    model=model,
    model_name="ion_vm",
    model_version="v1.2.0",
    model_type="ion_vm",
    feature_names=["energy_keV", "dose_cm2", ...],
    config=config
)

print(f"ONNX model saved to: {metadata.onnx_file_path}")

# Load and run inference
runtime = ONNXInferenceRuntime(metadata.onnx_file_path)

features = np.array([[30.0, 5e15, 7.0, 0.0, 1.02, 95.2, 1.2e6]])
prediction = runtime.predict_single(features[0])
print(f"Prediction: {prediction:.2f} Ω/□")

# Optimize ONNX model
optimized_path = exporter.optimize_onnx_model(metadata.onnx_file_path)

# Quantize to INT8
quantized_path = exporter.quantize_onnx_model(metadata.onnx_file_path)
```

---

## Usage Examples

### End-to-End Ion Implant Workflow

```python
import time
from app.spc import IonImplantMonitor, IonParameter
from app.vm import IonVirtualMetrologyModel, IonVMFeatures
from app.mlops import FeatureStore, ModelDriftMonitor

# 1. Initialize components
spc_monitor = IonImplantMonitor()
vm_model = IonVirtualMetrologyModel()
feature_store = FeatureStore()
drift_monitor = ModelDriftMonitor(model_name="ion_vm", feature_names=[...])

# 2. Start recipe
recipe_id = "ION_RECIPE_B11_30KEV"
spc_monitor.start_recipe(recipe_id)

# 3. Process wafer
wafer_id = "LOT_123_WAFER_05"

# Collect telemetry during run
telemetry = {
    IonParameter.BEAM_CURRENT_MA: 1.03,
    IonParameter.DOSE_UNIFORMITY_PCT: 94.5,
    IonParameter.SOURCE_PRESSURE_MTORR: 2.2e-5,
    IonParameter.ANALYZER_FIELD_GAUSS: 1518,
}

# 4. SPC monitoring
spc_alerts = spc_monitor.update(telemetry, timestamp=time.time())

if spc_alerts:
    print(f"⚠️  SPC Alerts detected: {len(spc_alerts)}")
    for alert in spc_alerts:
        print(f"  - {alert.parameter}: {alert.message}")

# 5. Virtual Metrology prediction
vm_features = IonVMFeatures(
    ion_species="B",
    energy_keV=30.0,
    dose_cm2=5e15,
    tilt_angle_deg=7.0,
    twist_angle_deg=0.0,
    beam_current_mA=telemetry[IonParameter.BEAM_CURRENT_MA],
    dose_uniformity_pct=telemetry[IonParameter.DOSE_UNIFORMITY_PCT]
)

vm_prediction = vm_model.predict(vm_features)

print(f"📊 VM Prediction:")
print(f"  Sheet Resistance: {vm_prediction.sheet_resistance_ohm_sq:.2f} Ω/□")
print(f"  Uncertainty: ±{vm_prediction.uncertainty_ohm_sq:.2f} Ω/□")
print(f"  Quality Score: {vm_prediction.quality_score}")

# 6. Store features
feature_store.write_online(
    entity_id=f"{recipe_id}_{wafer_id}",
    features={
        "ion_species_code": 5,
        "energy_keV": vm_features.energy_keV,
        "dose_cm2": vm_features.dose_cm2,
        # ... all features
    },
    feature_group="ion_implant_features"
)

# 7. Monitor drift (when ground truth available)
ground_truth = 152.3  # Actual measured Rs after metrology

drift_alerts = drift_monitor.update(
    features=np.array([30.0, 5e15, 7.0, ...]),
    prediction=vm_prediction.sheet_resistance_ohm_sq,
    ground_truth=ground_truth
)

if drift_alerts:
    print(f"🔄 Drift alerts: {len(drift_alerts)}")
```

### Model Training and Deployment

```python
from sklearn.ensemble import RandomForestRegressor
from app.mlops import FeatureStore, ModelRegistry, ModelMetadata, ONNXExporter

# 1. Get training data from feature store
feature_store = FeatureStore()
dataset = feature_store.get_training_dataset(
    feature_groups=["ion_implant_features"],
    start_date="2024-01-01",
    end_date="2025-01-01"
)

# 2. Prepare training data
X_train, y_train = prepare_data(dataset)  # Your data prep logic

# 3. Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# 4. Evaluate
r2_train = model.score(X_train, y_train)
r2_val = model.score(X_val, y_val)

# 5. Register model
registry = ModelRegistry()
metadata = ModelMetadata(
    model_name="ion_vm",
    model_version="v1.3.0",
    model_type="ion_vm",
    stage=ModelStage.DEVELOPMENT,
    training_date=datetime.now().isoformat(),
    training_samples=len(X_train),
    training_r2_score=r2_train,
    validation_r2_score=r2_val,
    model_file_path="",  # Set by registry
    model_file_hash=""
)

registry.register_model("./models/ion_vm_v1.3.0.pkl", metadata)

# 6. Export to ONNX
exporter = ONNXExporter()
onnx_metadata = exporter.export_sklearn_model(
    model=model,
    model_name="ion_vm",
    model_version="v1.3.0",
    model_type="ion_vm",
    feature_names=[...],
    config=ONNXExportConfig(validate=True, test_inputs=X_test)
)

# 7. Deploy to staging
registry.promote_model("ion_vm", "v1.3.0", ModelStage.STAGING)

# 8. After validation, deploy to production
registry.deploy_model(
    model_name="ion_vm",
    version="v1.3.0",
    strategy=DeploymentStrategy.CANARY,  # 5% traffic initially
    deployed_by="ml-engineer@fab.com"
)
```

---

## API Reference

### SPC Charts

**XbarRChart**
- `__init__(parameter_name, subgroup_size=5, window_size=25)`
- `update(values: List[float], timestamp: float)`
- `get_alerts() -> List[SPCAlert]`
- `get_statistics() -> Dict`

**EWMAChart**
- `__init__(parameter_name, lambda_weight=0.2, L=3.0, window_size=100)`
- `update(value: float, timestamp: float)`
- `get_alerts() -> List[SPCAlert]`

**CUSUMChart**
- `__init__(parameter_name, k=0.5, h=5.0, window_size=100)`
- `update(value: float, timestamp: float)`
- `get_alerts() -> List[SPCAlert]`

### Process Monitors

**IonImplantMonitor**
- `__init__(config: Optional[Dict] = None)`
- `start_recipe(recipe_id: str)`
- `update(measurements: Dict[IonParameter, float], timestamp: float) -> List[SPCAlert]`
- `get_summary() -> Dict`

**RTPMonitor**
- `__init__(config: Optional[Dict] = None)`
- Similar interface to IonImplantMonitor

### Virtual Metrology

**IonVirtualMetrologyModel**
- `predict(features: IonVMFeatures) -> IonVMPrediction`

**RTPVirtualMetrologyModel**
- `predict(features: RTPVMFeatures) -> RTPVMPrediction`

### MLOps

**FeatureStore**
- `register_feature_group(feature_group: FeatureGroup)`
- `write_online(entity_id: str, features: Dict, feature_group: str)`
- `read_online(entity_id: str, feature_group: str) -> Optional[Dict]`
- `write_offline(feature_vectors: List[FeatureVector], feature_group: str)`
- `read_offline(feature_group: str, start_date, end_date) -> List[FeatureVector]`

**ModelRegistry**
- `register_model(model_file_path: str, metadata: ModelMetadata) -> str`
- `get_model(model_name: str, version, stage) -> Optional[ModelMetadata]`
- `promote_model(model_name: str, version: str, target_stage: ModelStage)`
- `deploy_model(model_name: str, version: str, strategy: DeploymentStrategy)`
- `compare_models(model1_name, model1_version, model2_name, model2_version) -> Dict`

**DriftMonitor**
- `update(features: np.ndarray, prediction: float, ground_truth: Optional[float]) -> List[DriftAlert]`
- `get_drift_summary() -> Dict`

**ONNXExporter**
- `export_sklearn_model(...) -> ONNXMetadata`
- `export_xgboost_model(...) -> ONNXMetadata`
- `export_pytorch_model(...) -> ONNXMetadata`
- `optimize_onnx_model(onnx_path: str) -> str`
- `quantize_onnx_model(onnx_path: str) -> str`

---

## Deployment Guide

### Prerequisites

```bash
# Core dependencies
pip install numpy scipy scikit-learn

# ONNX support
pip install onnx onnxruntime skl2onnx onnxmltools

# Optional (for PyTorch models)
pip install torch
```

### Production Considerations

**Feature Store:**
- Replace JSON files with Redis for online serving
- Replace Pickle with Parquet/Delta Lake for offline storage
- Add feature versioning and lineage tracking
- Implement feature monitoring and data quality checks

**Model Registry:**
- Integrate with MLflow or Kubeflow for enterprise deployment
- Add A/B testing framework for model comparison
- Implement automated rollback on performance degradation
- Add model explainability (SHAP, LIME)

**Drift Monitoring:**
- Set up automated alerts (email, Slack, PagerDuty)
- Create drift dashboards with Grafana/Tableau
- Implement automated retraining pipelines
- Add concept drift detection with adversarial validation

**SPC Monitoring:**
- Integrate with real-time SCADA systems
- Add dashboards for operators (Grafana, custom UI)
- Implement automated corrective actions
- Add shift reports and trend analysis

**ONNX Deployment:**
- Deploy to ONNX Runtime Server for high-throughput inference
- Use TensorRT for GPU acceleration
- Implement model serving with Triton Inference Server
- Add request batching and model caching

### Performance Metrics

**SPC:**
- Alert latency: <100ms per measurement
- Throughput: >1000 measurements/sec per monitor
- Memory: ~10MB per chart (100-point history)

**VM:**
- Inference time: <10ms per prediction (ONNX)
- Throughput: >100 predictions/sec
- Memory: ~50MB per model

**MLOps:**
- Feature read latency: <5ms (online), <100ms (offline)
- Drift detection: <50ms per sample
- Model registry ops: <200ms

---

## Testing

Run unit tests:

```bash
# All tests
pytest services/process_control/tests/unit/

# SPC tests only
pytest services/process_control/tests/unit/spc/

# VM tests only
pytest services/process_control/tests/unit/vm/

# MLOps tests only
pytest services/process_control/tests/unit/mlops/
```

---

## Support and Contributions

For issues, questions, or contributions:
- File issues in project repository
- Contact: process-control-team@fab.com
- Documentation: [Internal Wiki](wiki.fab.com/spc-vm-ml)

---

## License

Proprietary - Internal Use Only
Copyright 2025 SPECTRA Lab

---

## Changelog

### v1.0.0 (2025-01-09)
- Initial release
- SPC charts: X-bar/R, EWMA, CUSUM
- Western Electric rules implementation
- Ion and RTP process monitors
- RCA playbooks with 9 failure modes
- Ion VM and RTP VM models
- Feature store with online/offline serving
- Model registry with 4 deployment strategies
- Drift monitoring (PSI, KS, JSD)
- ONNX export for scikit-learn, XGBoost, PyTorch
- Comprehensive documentation and examples
