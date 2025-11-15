# CVD Tool Drivers & Hardware-in-Loop Simulators

Vendor-agnostic CVD tool abstraction layer with physics-based simulation capabilities.

## Overview

This package provides a comprehensive framework for controlling CVD equipment and simulating CVD processes without hardware:

- **CVDTool Protocol**: Standard interface for all CVD tools (real and simulated)
- **Driver Families**: Vendor-agnostic implementations for major CVD technologies
- **HIL Simulator**: Physics-based process simulation with realistic metrics
- **Communication Adapters**: Support for SCPI, OPC-UA, and SECS-II protocols

## Architecture

```
CVDTool (Protocol)
    ├── HILCVDSimulator (Physics-based simulator)
    ├── ThermalCVDDriverBase
    │   ├── APCVDDriver
    │   ├── LPCVDDriver
    │   └── UHVCVDDriver
    ├── PlasmaCVDDriverBase
    │   ├── PECVDDriver
    │   ├── HDPCVDDriver
    │   ├── MPCVDDriver
    │   └── RPCVDDriver
    └── Specialty Drivers
        ├── MOCVDDriver
        └── AACVDDriver
```

## Quick Start

### 1. Using the HIL Simulator

```python
from app.drivers import HILCVDSimulator, PhysicsConfig

# Create simulator with custom physics
physics = PhysicsConfig(
    activation_energy_kj_mol=120.0,
    intrinsic_stress_mpa=-250.0,
    base_adhesion_score=85.0,
)

simulator = HILCVDSimulator(
    tool_id="SIM-LPCVD-01",
    mode="LPCVD",
    physics_config=physics,
)

# Connect and configure
await simulator.connect()
await simulator.configure(recipe)

# Start run and stream telemetry
await simulator.start_run(run_id)

async for telemetry in simulator.stream_telemetry(run_id, interval_sec=1.0):
    print(f"Temp: {telemetry.measurements[TelemetryType.TEMPERATURE]}°C")
    print(f"Thickness: {telemetry.thickness_nm} nm")
    print(f"Stress: {telemetry.stress_mpa} MPa")
```

### 2. Using a Real Driver (PECVD)

```python
from app.drivers import PECVDDriver

# Create driver
driver = PECVDDriver(
    tool_id="PECVD-FAB1-01",
    host="192.168.1.100",
    port=5025,
    vendor="Applied Materials",
    model="Centura",
)

# Connect and execute
await driver.connect()

# Get capabilities
caps = await driver.get_capabilities()
print(f"Max RF Power: {caps.max_rf_power_w}W")
print(f"Supported modes: {caps.supported_modes}")

# Load recipe and start
await driver.configure(recipe)
await driver.start_run(run_id)

# Monitor process
status = await driver.get_status()
print(f"State: {status.state}")
print(f"Chamber Temp: {status.chamber_temp_c}°C")
```

### 3. Using MOCVD Driver

```python
from app.drivers import MOCVDDriver

driver = MOCVDDriver(
    tool_id="MOCVD-R&D-01",
    host="192.168.1.150",
    port=4840,  # OPC-UA
    vendor="Aixtron",
    model="AIX 2800G4",
)

await driver.connect()

# MOCVD-specific telemetry includes V/III ratio
async for telemetry in driver.stream_telemetry(run_id):
    v_iii_ratio = telemetry.raw_data.get('v_iii_ratio')
    print(f"V/III Ratio: {v_iii_ratio}")
    print(f"Growth Rate: {telemetry.deposition_rate_nm_min} nm/min")
```

## CVDTool Protocol Interface

All drivers and simulators implement this standard interface:

```python
class CVDTool(Protocol):
    async def get_capabilities() -> ToolCapabilities
    async def connect() -> None
    async def disconnect() -> None
    async def configure(recipe) -> None
    async def start_run(cvd_run_id: UUID) -> None
    async def stop_run(cvd_run_id: UUID) -> None
    async def pause_run(cvd_run_id: UUID) -> None
    async def resume_run(cvd_run_id: UUID) -> None
    async def abort_run(cvd_run_id: UUID) -> None
    async def get_status(cvd_run_id: UUID) -> ToolStatus
    async def stream_telemetry(cvd_run_id: UUID) -> AsyncIterator[CVDTelemetry]
    async def get_alarms() -> List[Dict[str, Any]]
    async def clear_alarms() -> None
    async def run_diagnostics() -> Dict[str, Any]
```

## Driver Families

### Thermal CVD

**APCVDDriver** - Atmospheric Pressure CVD
- Pressure: 500-1000 Torr
- Temperature: 400-1200°C
- Applications: Oxides, nitrides
- Communication: SCPI

**LPCVDDriver** - Low Pressure CVD
- Pressure: 0.1-10 Torr
- Temperature: 400-1200°C
- Batch processing (up to 100 wafers)
- Excellent uniformity and conformality
- Communication: OPC-UA

**UHVCVDDriver** - Ultra-High Vacuum CVD
- Pressure: <1e-6 Torr
- Temperature: 400-800°C
- Single wafer processing
- Ultra-clean deposition
- Communication: SECS-II

### Plasma CVD

**PECVDDriver** - Plasma-Enhanced CVD
- RF: 13.56 MHz
- Pressure: 0.1-10 Torr
- Temperature: 150-400°C
- Standard dielectrics (SiO₂, SiN)
- Communication: SCPI

**HDPCVDDriver** - High-Density Plasma CVD
- Inductively coupled plasma
- Excellent gap fill
- Lower ion damage
- Communication: SECS-II

**MPCVDDriver** - Microwave Plasma CVD
- Frequency: 2.45 GHz
- Temperature: 400-1200°C
- Diamond and DLC films
- Communication: OPC-UA

**RPCVDDriver** - Remote Plasma CVD
- Downstream plasma
- Low substrate damage
- Low-k dielectrics
- Communication: SCPI

### Specialty CVD

**MOCVDDriver** - Metal-Organic CVD
- Epitaxial III-V compounds (GaN, GaAs, InP)
- Pressure: 50-760 Torr
- Temperature: 500-1200°C
- V/III ratio control
- Communication: OPC-UA

**AACVDDriver** - Aerosol-Assisted CVD
- Nanomaterials and complex oxides
- Liquid precursor aerosolization
- Transparent conducting oxides
- Communication: SCPI

## HIL Simulator

### Physics-Based Modeling

The HIL simulator uses realistic physics models:

#### 1. Deposition Rate (Arrhenius Kinetics)
```
rate = A * exp(-Ea/RT) * P^n * (1 + k_flow * flow) * (1 + k_power * power)
```

#### 2. Film Stress
```
σ_total = σ_intrinsic + σ_thermal + σ_gradient

σ_thermal = [E/(1-ν)] * Δα * ΔT
```

#### 3. Adhesion Score
```
adhesion = base_adhesion * surface_quality * stress_penalty

stress_penalty = 1 / (1 + k_stress * |σ|)
```

#### 4. Thickness Distribution
- Radial non-uniformity (center-to-edge)
- Conformality ratio (step coverage)
- WIW and WTW uniformity

### Fault Injection

For FDC/SPC testing:

```python
from app.drivers import FaultInjectionConfig

faults = FaultInjectionConfig(
    enabled=True,
    temp_spike_probability=0.001,  # Per second
    temp_spike_magnitude_c=50.0,
    pressure_leak_probability=0.0005,
    flow_blockage_probability=0.0002,
)

simulator = HILCVDSimulator(
    tool_id="SIM-01",
    mode="LPCVD",
    fault_config=faults,
)
```

## Communication Adapters

### SCPI/VISA

For text-based command instruments:

```python
from app.drivers import SCPIAdapter

adapter = SCPIAdapter(host="192.168.1.100", port=5025)
await adapter.connect()

# Query commands
temp = await adapter.query("SYSTem:TEMPerature?")
status = await adapter.query("STATus:OPERation?")

# Write commands
await adapter.write("PROCess:STARt")
```

### OPC-UA

For modern industrial automation:

```python
from app.drivers import OPCUAAdapter

adapter = OPCUAAdapter(endpoint_url="opc.tcp://192.168.1.100:4840")
await adapter.connect()

# Read nodes
temp = await adapter.read_node("ns=2;s=Temperature")

# Write nodes
await adapter.write_node("ns=2;s=Setpoint", 850.0)

# Call methods
await adapter.call_method("ns=2;s=CVDTool", "ns=2;s=Start", "recipe123")
```

### SECS-II/GEM

For semiconductor manufacturing equipment:

```python
from app.drivers import SECS2Adapter

adapter = SECS2Adapter(host="192.168.1.100", port=5000, device_id=0)
await adapter.connect()

# Standard SECS messages
info = await adapter.send_s1f1_are_you_there()
status = await adapter.send_s1f3_get_status([1, 2, 3])

# Remote commands
ack = await adapter.send_s2f41_remote_command("START", [])

# Wait for events
event = await adapter.wait_for_event(ceid=101)
```

## Advanced Features

### Multi-Zone Temperature Control

```python
# Telemetry includes zone-specific data
async for telemetry in simulator.stream_telemetry(run_id):
    if telemetry.zone_temps_c:
        for zone, temp in telemetry.zone_temps_c.items():
            print(f"{zone}: {temp}°C")
```

### In-Situ Metrology

```python
# Real-time thickness and stress monitoring
async for telemetry in simulator.stream_telemetry(run_id):
    print(f"Thickness: {telemetry.thickness_nm} nm")
    print(f"Deposition Rate: {telemetry.deposition_rate_nm_min} nm/min")
    print(f"Stress: {telemetry.stress_mpa} MPa")
    print(f"Reflectance: {telemetry.reflectance_pct}%")
```

### Recipe Step Control

```python
# Monitor multi-step recipes
async for telemetry in simulator.stream_telemetry(run_id):
    print(f"Step {telemetry.step_number}/{status.total_steps}: {telemetry.step_name}")
    print(f"Elapsed: {telemetry.elapsed_time_sec}s")
    print(f"Remaining: {status.estimated_remaining_sec}s")
```

## Extending Drivers

### Creating a Custom Driver

Inherit from the appropriate base class:

```python
from app.drivers import ThermalCVDDriverBase

class MyCustomLPCVDDriver(ThermalCVDDriverBase):
    def __init__(self, tool_id: str, host: str, port: int):
        super().__init__(
            tool_id=tool_id,
            host=host,
            port=port,
            vendor="MyCompany",
            model="CustomLPCVD",
            mode="LPCVD",
        )

    async def _establish_connection(self):
        # Implement vendor-specific connection
        pass

    async def _send_recipe(self, recipe):
        # Implement vendor-specific recipe loading
        pass

    # ... implement other abstract methods
```

### Adding New Physics Models

Customize the HIL simulator:

```python
class CustomPhysicsSimulator(HILCVDSimulator):
    def _calculate_deposition_rate(self, temp_c, pressure_torr, flow_rate_sccm, rf_power_w):
        # Override with custom model
        custom_rate = self._my_custom_kinetics(temp_c, pressure_torr)
        return custom_rate

    def _calculate_film_stress(self, thickness_nm, deposition_temp_c):
        # Override stress model
        return self._advanced_stress_model(thickness_nm, deposition_temp_c)
```

## Testing

### Unit Tests

```python
import pytest
from app.drivers import HILCVDSimulator

@pytest.mark.asyncio
async def test_simulator_connection():
    sim = HILCVDSimulator(tool_id="TEST-01", mode="LPCVD")

    await sim.connect()
    assert sim.state == ToolState.IDLE

    caps = await sim.get_capabilities()
    assert "LPCVD" in caps.supported_modes

    await sim.disconnect()
    assert sim.state == ToolState.OFFLINE
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_deposition_cycle():
    sim = HILCVDSimulator(tool_id="TEST-01", mode="LPCVD")

    await sim.connect()
    await sim.configure(mock_recipe)

    run_id = uuid4()
    await sim.start_run(run_id)

    # Collect telemetry
    telemetry_samples = []
    async for telem in sim.stream_telemetry(run_id, interval_sec=0.1):
        telemetry_samples.append(telem)
        if len(telemetry_samples) >= 10:
            break

    await sim.stop_run(run_id)

    # Verify physics
    assert telemetry_samples[0].thickness_nm < telemetry_samples[-1].thickness_nm
    assert all(t.stress_mpa < 0 for t in telemetry_samples)  # Compressive
```

## Performance Considerations

- **Telemetry Interval**: 1 second is typical for SPC. Use 0.1s for critical process control.
- **Connection Pooling**: Reuse connections for multiple runs on same tool.
- **Async I/O**: All drivers use async/await for non-blocking operation.
- **Error Handling**: All methods raise `ToolError` with error codes for diagnostics.

## Troubleshooting

### Connection Issues

```python
try:
    await driver.connect()
except ToolError as e:
    if e.error_code == "CONN_FAILED":
        # Check network, firewall, tool power
        pass
```

### Recipe Validation Errors

```python
try:
    await driver.configure(recipe)
except ToolError as e:
    if e.error_code == "CONFIG_FAILED":
        # Check recipe parameters against capabilities
        caps = await driver.get_capabilities()
        print(f"Max temp: {caps.max_temp_c}°C")
```

### Process Faults

```python
status = await driver.get_status()
if status.state == ToolState.ERROR:
    alarms = await driver.get_alarms()
    for alarm in alarms:
        logger.error(f"Alarm: {alarm['message']}")

    # Clear after troubleshooting
    await driver.clear_alarms()
```

## Future Enhancements

- [ ] Add more CVD variants (ALD, Pulsed CVD, HWCVD)
- [ ] Implement digital twin synchronization
- [ ] Add machine learning for adaptive process control
- [ ] Support for multi-chamber cluster tools
- [ ] Real-time FDC/R2R integration
- [ ] OES (Optical Emission Spectroscopy) endpoint detection

## References

- SEMI Standards (E5, E30, E37, E87, E94, E120)
- SCPI Specification v1999.0
- OPC UA Specification Part 1-14
- CVD Physics: Handbook of Thin Film Deposition (Seshan)
- Stress in Films: Freund & Suresh, Thin Film Materials

## License

Internal use only - SPECTRA Lab
