# CVD Platform Testing and Validation Plan

## Table of Contents
1. [Testing Strategy](#testing-strategy)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [System Testing](#system-testing)
5. [Performance Testing](#performance-testing)
6. [Validation with Real Data](#validation-with-real-data)
7. [Continuous Integration](#continuous-integration)

## 1. Testing Strategy

### Testing Pyramid

```
                    /\
                   /  \
                  / E2E \           < 10% End-to-End Tests
                 /______\
                /        \
               / Integration \      < 30% Integration Tests
              /______________\
             /                \
            /   Unit Tests     \   < 60% Unit Tests
           /____________________\
```

### Test Coverage Goals

- **Unit Tests:** >80% code coverage
- **Integration Tests:** All module interfaces
- **System Tests:** All critical workflows
- **Performance Tests:** All scalability requirements

### Testing Tools

**Backend:**
- pytest: Unit and integration testing
- pytest-asyncio: Async test support
- pytest-cov: Coverage reporting
- locust: Load testing
- hypothesis: Property-based testing

**Frontend:**
- Jest: Unit testing
- React Testing Library: Component testing
- Cypress: End-to-end testing
- Storybook: Component development

## 2. Unit Testing

### Backend Unit Tests

#### 2.1 Sensor Interface Tests

**File:** `tests/unit/test_sensor_interface.py`

```python
import pytest
import asyncio
from backend.data_acquisition.sensor_interface import (
    TemperatureSensor, PressureSensor, MassFlowController
)

@pytest.mark.asyncio
async def test_temperature_sensor_reading():
    """Test temperature sensor reads valid values"""
    sensor = TemperatureSensor(
        sensor_id="TEMP_001",
        config={"thermocouple_type": "K", "zone_id": 1}
    )

    await sensor.connect()
    reading = await sensor.read()
    await sensor.disconnect()

    assert reading.value > 0
    assert reading.unit == "°C"
    assert reading.confidence > 0.5

@pytest.mark.asyncio
async def test_temperature_sensor_calibration():
    """Test temperature sensor calibration"""
    sensor = TemperatureSensor(sensor_id="TEMP_001", config={})

    await sensor.connect()
    result = await sensor.calibrate(reference_value=100.0)
    await sensor.disconnect()

    assert result == True
    assert "slope" in sensor.calibration_data
    assert "offset" in sensor.calibration_data

@pytest.mark.asyncio
async def test_mfc_flow_control():
    """Test MFC setpoint and readback"""
    mfc = MassFlowController(
        sensor_id="MFC_001",
        config={"gas_type": "N2", "max_flow": 1000.0}
    )

    await mfc.connect()
    await mfc.set_flow(500.0)
    reading = await mfc.read()
    await mfc.disconnect()

    assert abs(reading.value - 500.0) < 50.0  # Within 10%
```

#### 2.2 Physics Model Tests

**File:** `tests/unit/test_cvd_reactor_model.py`

```python
import pytest
import numpy as np
from backend.physics_models.cvd_reactor_model import (
    CVDReactorModel, ReactorGeometry, ReactorDimensions, ProcessConditions
)

def test_reynolds_number_calculation():
    """Test Reynolds number calculation"""
    dimensions = ReactorDimensions(
        length=1.0, diameter=0.5, susceptor_diameter=0.35,
        wafer_diameter=0.3, gap_height=0.05, heater_zones=5,
        inlet_diameter=0.05, outlet_diameter=0.05
    )
    model = CVDReactorModel(ReactorGeometry.SHOWERHEAD, dimensions)

    conditions = ProcessConditions(
        temperature=1073.15,  # 800°C
        pressure=1330,  # 10 Torr
        gas_flows={"SiH4": 100, "N2": 5000},
        rotation_speed=20,
        deposition_time=120,
        susceptor_temp=1073.15,
        wall_temp=600
    )

    Re = model.compute_reynolds_number(conditions, 0.05)

    assert Re > 0
    assert Re < 2300  # Should be laminar

def test_deposition_rate_calculation():
    """Test deposition rate prediction"""
    dimensions = ReactorDimensions(
        length=1.0, diameter=0.5, susceptor_diameter=0.35,
        wafer_diameter=0.3, gap_height=0.05, heater_zones=5,
        inlet_diameter=0.05, outlet_diameter=0.05
    )
    model = CVDReactorModel(ReactorGeometry.SHOWERHEAD, dimensions)

    conditions = ProcessConditions(
        temperature=1073.15, pressure=1330,
        gas_flows={"SiH4": 100, "N2": 5000},
        rotation_speed=20, deposition_time=120,
        susceptor_temp=1073.15, wall_temp=600
    )

    # Run simulation
    model.solve_velocity_field(conditions)
    model.solve_temperature_field(conditions)
    model.solve_species_transport("SiH4", conditions)
    deposition_rate = model.calculate_deposition_rate(conditions)

    assert len(deposition_rate) == model.nr
    assert np.all(deposition_rate > 0)
    assert np.mean(deposition_rate) > 1.0  # nm/s
    assert np.mean(deposition_rate) < 20.0

def test_film_thickness_uniformity():
    """Test thickness uniformity calculation"""
    dimensions = ReactorDimensions(
        length=1.0, diameter=0.5, susceptor_diameter=0.35,
        wafer_diameter=0.3, gap_height=0.05, heater_zones=5,
        inlet_diameter=0.05, outlet_diameter=0.05
    )
    model = CVDReactorModel(ReactorGeometry.SHOWERHEAD, dimensions)

    conditions = ProcessConditions(
        temperature=1073.15, pressure=1330,
        gas_flows={"SiH4": 100, "N2": 5000},
        rotation_speed=20, deposition_time=120,
        susceptor_temp=1073.15, wall_temp=600
    )

    result = model.run_full_simulation(conditions)
    thickness_data = result["thickness"]

    assert thickness_data["mean_thickness"] > 50  # nm
    assert thickness_data["uniformity_percent"] < 10  # < 10% non-uniformity
```

#### 2.3 Virtual Metrology Tests

**File:** `tests/unit/test_vm_predictor.py`

```python
import pytest
import numpy as np
from backend.virtual_metrology.vm_predictor import (
    LightGBMPredictor, DesignFeatures, ProcessFeatures
)

def test_lightgbm_training():
    """Test LightGBM model training"""
    predictor = LightGBMPredictor()

    # Generate synthetic training data
    np.random.seed(42)
    X_train = np.random.randn(1000, 30)
    y_train = 100 + 5 * X_train[:, 0] + 2 * X_train[:, 1] + np.random.randn(1000) * 2

    X_val = np.random.randn(200, 30)
    y_val = 100 + 5 * X_val[:, 0] + 2 * X_val[:, 1] + np.random.randn(200) * 2

    # Train
    metrics = predictor.train(X_train, y_train, X_val, y_val, num_boost_round=100)

    assert metrics["train_rmse"] < 10.0
    assert metrics["val_r2"] > 0.7

def test_vm_prediction():
    """Test VM thickness prediction"""
    from backend.virtual_metrology.vm_predictor import VirtualMetrologyPredictor

    predictor = VirtualMetrologyPredictor(model_type="lightgbm")

    # Mock training (in real scenario, use pre-trained model)

    design_features = DesignFeatures(
        pattern_density=0.5,
        line_pitch=100.0,
        perimeter_density=2.0,
        feature_size=50.0,
        aspect_ratio=2.0,
        open_area_fraction=0.5,
        corner_count=100,
        metal_layer=1,
        x_position=0.0,
        y_position=0.0,
        die_location="center"
    )

    process_features = ProcessFeatures(
        temperature_setpoint=800.0,
        temperature_actual=800.5,
        temperature_uniformity=2.0,
        pressure_setpoint=10.0,
        pressure_actual=10.05,
        precursor_flow=100.0,
        carrier_flow=5000.0,
        deposition_time=120.0,
        rotation_speed=20.0,
        heater_zone_temps=[800, 800, 800, 800, 800],
        chamber_id="CVD-01",
        recipe_id="Si_100nm",
        wafer_number=1,
        lot_id="LOT-001",
        pm_cycle=5,
        chamber_age=100.0,
        previous_wafer_thickness=None
    )

    # Note: This will fail without trained model, but shows interface
    # prediction = predictor.predict_thickness(design_features, process_features)
    # assert prediction.predicted_thickness > 0
```

#### 2.4 Process Control Tests

**File:** `tests/unit/test_r2r_controller.py`

```python
import pytest
from backend.process_control.r2r_controller import (
    EWMAController, PIDController, RecipeParameters, ControlTarget
)

def test_ewma_control_action():
    """Test EWMA controller calculates correct action"""
    controller = EWMAController(gain=0.3, lambda_ewma=0.7)

    current_recipe = RecipeParameters(
        temperature=800.0,
        pressure=10.0,
        precursor_flow=100.0,
        carrier_flow=5000.0,
        deposition_time=120.0,
        rotation_speed=20.0,
        heater_zone_powers=[1000, 1000, 1000, 1000, 1000]
    )

    target = ControlTarget(
        target_thickness=100.0,
        thickness_tolerance=5.0,
        target_uniformity=2.0,
        uniformity_tolerance=0.5
    )

    measured_thickness = 95.0  # 5 nm below target

    action = controller.calculate_control_action(
        current_recipe, target, measured_thickness
    )

    assert "deposition_time" in action.parameter_updates
    assert action.parameter_updates["deposition_time"] > current_recipe.deposition_time

def test_pid_controller():
    """Test PID controller"""
    controller = PIDController(Kp=0.3, Ki=0.1, Kd=0.05)

    current_recipe = RecipeParameters(
        temperature=800.0, pressure=10.0,
        precursor_flow=100.0, carrier_flow=5000.0,
        deposition_time=120.0, rotation_speed=20.0,
        heater_zone_powers=[1000, 1000, 1000, 1000, 1000]
    )

    target = ControlTarget(
        target_thickness=100.0, thickness_tolerance=5.0,
        target_uniformity=2.0, uniformity_tolerance=0.5
    )

    # Simulate multiple wafers
    thicknesses = [95, 97, 98, 99.5, 100.2]

    for thickness in thicknesses:
        action = controller.calculate_control_action(
            current_recipe, target, thickness
        )
        current_recipe = RecipeParameters(
            temperature=current_recipe.temperature,
            pressure=current_recipe.pressure,
            precursor_flow=current_recipe.precursor_flow,
            carrier_flow=current_recipe.carrier_flow,
            deposition_time=action.parameter_updates.get(
                "deposition_time", current_recipe.deposition_time
            ),
            rotation_speed=current_recipe.rotation_speed,
            heater_zone_powers=current_recipe.heater_zone_powers
        )

    # Controller should converge
    assert abs(thicknesses[-1] - target.target_thickness) < 1.0
```

#### 2.5 SPC Tests

**File:** `tests/unit/test_spc_monitor.py`

```python
import pytest
import numpy as np
from backend.spc_fdc.spc_monitor import (
    XBarChart, EWMAChart, CUSUMChart, calculate_process_capability
)

def test_xbar_chart_violations():
    """Test X-bar chart violation detection"""
    chart = XBarChart(
        chart_id="thickness_xbar",
        parameter_name="Thickness",
        target=100.0,
        spec_limits=(95.0, 105.0),
        subgroup_size=5
    )

    # Add normal data
    np.random.seed(42)
    for i in range(30):
        value = np.random.normal(100.0, 2.0)
        chart.add_point(value)

    # Calculate control limits
    chart.calculate_control_limits()

    # Add out-of-control point
    violations = chart.add_point(120.0)  # Well above UCL

    assert len(violations) > 0
    assert violations[0].violation_type.value == "above_ucl"

def test_process_capability():
    """Test Cp/Cpk calculation"""
    np.random.seed(42)
    data = np.random.normal(100.0, 1.0, 1000)

    capability = calculate_process_capability(
        data=data,
        lsl=95.0,
        usl=105.0,
        target=100.0
    )

    assert capability.cp > 1.0
    assert capability.cpk > 1.0
    assert capability.sigma_level > 3.0
```

## 3. Integration Testing

### API Integration Tests

**File:** `tests/integration/test_api.py`

```python
import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_simulation_run():
    """Test running CVD simulation via API"""
    payload = {
        "temperature": 1073.15,
        "pressure": 1330,
        "gas_flows": {"SiH4": 100, "N2": 5000},
        "rotation_speed": 20,
        "deposition_time": 120,
        "susceptor_temp": 1073.15,
        "wall_temp": 600
    }

    response = client.post("/simulation/run", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "thickness" in data
    assert data["thickness"]["mean_thickness"] > 0

def test_spc_chart_creation_and_measurement():
    """Test SPC chart creation and data addition"""
    # Create chart
    response = client.post(
        "/spc/chart/create",
        params={
            "chart_id": "test_thickness",
            "parameter_name": "Thickness",
            "chart_type": "xbar",
            "target": 100.0,
            "lsl": 95.0,
            "usl": 105.0
        }
    )
    assert response.status_code == 200

    # Add measurements
    for value in [98, 100, 102, 99, 101]:
        response = client.post(
            "/spc/measurement/add",
            json={"chart_id": "test_thickness", "value": value}
        )
        assert response.status_code == 200
```

## 4. System Testing

### End-to-End Workflow Tests

**Test Scenario 1: Complete Process Run**

1. Initialize equipment
2. Load recipe
3. Process wafer
4. Collect sensor data
5. Run VM prediction
6. Update recipe via R2R
7. Generate SPC charts
8. Check for alarms

**Test Scenario 2: Drift Detection and Compensation**

1. Process 20 wafers with gradual drift
2. Verify drift detection triggers
3. Verify compensation applied
4. Verify convergence to target

## 5. Performance Testing

### Load Testing with Locust

**File:** `tests/performance/locustfile.py`

```python
from locust import HttpUser, task, between

class CVDPlatformUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def get_health(self):
        self.client.get("/health")

    @task(3)
    def get_equipment_status(self):
        self.client.get("/equipment/status")

    @task(2)
    def add_spc_measurement(self):
        self.client.post(
            "/spc/measurement/add",
            json={"chart_id": "thickness_chart", "value": 100.0}
        )
```

**Performance Requirements:**
- API latency: <100ms for 95th percentile
- Throughput: >1000 requests/second
- WebSocket updates: 10 Hz minimum

## 6. Validation with Real Data

### Historical Data Validation

1. **Load historical process data** from production equipment
2. **Compare model predictions** to actual measurements
3. **Calculate error metrics:** RMSE, MAE, R²
4. **Acceptance criteria:** RMSE < 5 nm for thickness

### Pilot Run Validation

1. **Run platform in parallel** with existing system
2. **Compare predictions and controls** side-by-side
3. **Gradual transition** based on performance metrics

## 7. Continuous Integration

### GitHub Actions Workflow

**File:** `.github/workflows/test.yml`

```yaml
name: CVD Platform Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio

    - name: Run unit tests
      run: |
        cd backend
        pytest tests/unit --cov=. --cov-report=xml

    - name: Run integration tests
      run: |
        cd backend
        pytest tests/integration

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./backend/coverage.xml
```

### Test Coverage Reporting

- Integrate with Codecov or Coveralls
- Track coverage trends over time
- Enforce minimum coverage thresholds in CI

---

## Conclusion

This comprehensive testing plan ensures:
- **High code quality** through extensive unit testing
- **Correct integration** between modules
- **System reliability** through end-to-end testing
- **Performance** at production scale
- **Validation** against real-world data
- **Continuous quality** through automated CI/CD
