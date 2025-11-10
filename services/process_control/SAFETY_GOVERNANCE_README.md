# Process Control Safety & Governance System

## Overview

This module implements comprehensive safety, calibration, and governance features for the Process Control system (Ion Implantation and RTP), ensuring compliant and auditable operations.

## Features Implemented

### 1. Safety Classification & SOP Gates

**Hazard Levels:**
- `LOW` - Routine operations, no special approvals required
- `MEDIUM` - Standard safety protocols
- `HIGH` - Requires dual approval before proceeding
- `CRITICAL` - Requires dual approval + additional oversight

**SOP Gates Defined:**
- **Ion Implantation Standard** (`HIGH` hazard level)
  - Hazards: High voltage (up to 500 kV), vacuum, ionizing radiation (X-rays), toxic/flammable gases
  - Required PPE: Safety glasses, lab coat, closed-toe shoes
  - Interlock checks: Chamber door, beam current, vacuum level, cooling water
  - Required calibrations: Beam current integrator, analyzer magnet probe, vacuum gauge

- **RTP High Temperature** (`HIGH` hazard level)
  - Hazards: High temperature (up to 1200°C), hot surfaces, thermal radiation, reactive gases
  - Required PPE: Heat-resistant gloves, safety glasses, lab coat
  - Interlock checks: Chamber door, temperature limits, cooling water, gas flows
  - Required calibrations: Pyrometer, thermocouple, pressure gauge, mass flow controllers

### 2. Calibration Management

**Calibration Statuses:**
- `VALID` - Calibration is current and valid
- `EXPIRING_SOON` - Within 30 days of expiry (warning)
- `EXPIRED` - Past expiry date (blocks run)
- `NOT_CALIBRATED` - No calibration record exists (blocks run)

**Instrument Types:**
- Ion Implantation:
  - Beam Current Integrator
  - Analyzer Magnet Probe
  - Vacuum Gauge

- RTP:
  - Pyrometer
  - Thermocouple
  - Pressure Gauge
  - Mass Flow Controller

**Calibration Lockout:**
If any required instrument has expired calibration, the system will:
1. Block run from starting
2. Return HTTP 403 with error code `CALIBRATION_EXPIRED`
3. Provide detailed information about expired instruments
4. Indicate days overdue and action required

### 3. Approval Workflows

**Approval Requirements:**
- `HIGH` and `CRITICAL` hazard levels require dual approval
- Both approvers must be different individuals
- Approvals are tracked with timestamps and e-signatures
- Approval requests create immutable audit records

**Approval States:**
- `PENDING` - Awaiting approval(s)
- `APPROVED` - Fully approved (all required approvers)
- `REJECTED` - Denied with reason

**API Guards:**
The `check_safety_approval()` dependency function ensures:
1. Approval exists for the run
2. Approval status is `APPROVED`
3. Required number of approvers have signed off
4. No run can proceed without proper approval for HIGH/CRITICAL processes

### 4. Uncertainty Budgets

**Ion Implantation Uncertainty:**
Components include:
- **Dose uncertainty:**
  - Current integrator accuracy (from calibration)
  - Scan uniformity (~2%)
  - Charge collection efficiency (~1%)

- **Range uncertainty:**
  - Energy uncertainty (~1%)
  - Species purity (~0.5%)
  - Tilt angle error

- **Straggle uncertainty:**
  - Energy spread (~10%)
  - Target composition (~5%)

**RTP Uncertainty:**
Components include:
- **Temperature uncertainty:**
  - Emissivity error (typically ±0.05)
  - Pyrometer calibration accuracy
  - Sensor lag (~2°C)
  - Spatial non-uniformity (~5°C)

- **Ramp rate uncertainty:**
  - Setpoint tracking error
  - Power control precision
  - Thermal inertia effects

All uncertainties reported as:
- Type A (statistical)
- Type B (systematic)
- Combined standard uncertainty
- Expanded uncertainty (k=2, 95% confidence)

### 5. Audit Trail & E-Signatures

**Audit Record Types:**
- `approval_request` - Safety approval submitted
- `approval_granted` - Approval decision (positive)
- `approval_rejected` - Approval decision (negative)
- `run_start` - Process run initiated
- `run_complete` - Process run completed successfully
- `run_abort` - Process run aborted
- `e_signature` - E-signature captured

**Features:**
- Immutable records (append-only)
- Blockchain-like integrity with previous record hashing
- Cryptographic e-signatures support
- Full user attribution
- Detailed event context

## API Endpoints

### Calibration Management

```
POST   /api/v1/safety/calibration/records        Create calibration record
GET    /api/v1/safety/calibration/records        List calibration records
GET    /api/v1/safety/calibration/records/{id}   Get specific calibration
POST   /api/v1/safety/calibration/check          Check calibration status
```

### Safety Approvals

```
POST   /api/v1/safety/approvals/request          Submit approval request
POST   /api/v1/safety/approvals/{id}/approve     Approve/reject request
GET    /api/v1/safety/approvals                  List approvals
GET    /api/v1/safety/approvals/{id}             Get specific approval
GET    /api/v1/safety/approvals/run/{run_id}     Get approval by run ID
POST   /api/v1/safety/approvals/run/{id}/check   Check approval for run
```

### SOP Gates

```
GET    /api/v1/safety/sop-gates                  List all SOP gates
GET    /api/v1/safety/sop-gates/{key}            Get specific SOP gate
```

### Uncertainty Budgets

```
POST   /api/v1/safety/uncertainty/ion-implant    Calculate ion implant uncertainty
POST   /api/v1/safety/uncertainty/rtp            Calculate RTP uncertainty
```

### Audit Trail

```
GET    /api/v1/safety/audit/records              List audit records
GET    /api/v1/safety/audit/records/{id}         Get specific audit record
POST   /api/v1/safety/audit/e-sign               Create e-signature record
```

### Compliance Summary

```
GET    /api/v1/safety/compliance/summary         Get overall compliance status
```

## Error Codes

The system uses consistent error codes for safety violations:

- `CALIBRATION_NOT_FOUND` - No calibration record exists
- `CALIBRATION_EXPIRED` - Instrument calibration has expired
- `APPROVAL_REQUIRED` - Safety approval required for this hazard level
- `APPROVAL_REJECTED` - Safety approval was rejected
- `APPROVAL_PENDING` - Approval request is still pending
- `INSUFFICIENT_APPROVALS` - Not all required approvals obtained

## Usage Examples

### 1. Check Calibration Before Run

```python
import requests
from uuid import UUID

instrument_ids = [
    "123e4567-e89b-12d3-a456-426614174000",  # Beam current integrator
    "223e4567-e89b-12d3-a456-426614174001",  # Analyzer magnet
]

response = requests.post(
    "http://localhost:8003/api/v1/safety/calibration/check",
    json={"instrument_ids": instrument_ids}
)

if response.status_code == 200:
    print("All calibrations valid")
    print(response.json())
elif response.status_code == 403:
    error = response.json()
    print(f"Calibration expired: {error['message']}")
    for expired in error["expired_instruments"]:
        print(f"  - {expired['name']}: {expired['days_overdue']} days overdue")
```

### 2. Request Safety Approval

```python
approval_request = {
    "run_id": "323e4567-e89b-12d3-a456-426614174002",
    "process_type": "ion_implant",
    "hazard_level": "HIGH",
    "requested_by": "john.doe@example.com",
    "hazards": [
        "High voltage operation (500 kV)",
        "X-ray generation"
    ],
    "mitigations": [
        "All interlocks verified",
        "Radiation monitoring active",
        "Personnel briefed on emergency procedures"
    ]
}

response = requests.post(
    "http://localhost:8003/api/v1/safety/approvals/request",
    json=approval_request
)

approval_id = response.json()["id"]
print(f"Approval requested: {approval_id}")
```

### 3. Approve Safety Request

```python
decision = {
    "approver": "jane.smith@example.com",
    "approved": True,
    "e_signature": "digital_signature_hash_here"
}

response = requests.post(
    f"http://localhost:8003/api/v1/safety/approvals/{approval_id}/approve",
    json=decision
)

approval = response.json()
if approval["is_fully_approved"]:
    print("Run approved and ready to proceed")
else:
    print("Awaiting second approval")
```

### 4. Calculate Uncertainty Budget

```python
uncertainty_request = {
    "energy_keV": 100.0,
    "dose_cm2": 1e15,
    "current_mA": 5.0,
    "instrument_ids": instrument_ids
}

response = requests.post(
    "http://localhost:8003/api/v1/safety/uncertainty/ion-implant",
    json=uncertainty_request
)

budget = response.json()
dose_unc = budget["dose_uncertainty"]
print(f"Dose: {dose_unc['nominal_value']:.2e} ± {dose_unc['expanded_uncertainty']:.2e} ions/cm²")
print(f"Contributors: {dose_unc['contributors']}")
```

### 5. View Compliance Summary

```python
response = requests.get("http://localhost:8003/api/v1/safety/compliance/summary")
summary = response.json()

print(f"Calibration Compliance: {summary['calibration']['compliance_rate']:.1f}%")
print(f"  Valid: {summary['calibration']['valid']}")
print(f"  Expired: {summary['calibration']['expired']}")
print(f"Approval Rate: {summary['approvals']['approval_rate']:.1f}%")
print(f"Overall Status: {summary['overall_status']}")
```

## Integration with Existing Endpoints

To integrate safety guards into existing run control endpoints, use the dependency functions:

```python
from app.safety import check_calibration_status, check_safety_approval, create_audit_record

@implant_router.post("/control/start")
async def start_implantation(
    run_id: UUID,
    target_dose_cm2: float,
    instrument_ids: List[UUID],
    user: str = "system"
):
    # Check calibrations (raises exception if expired)
    await check_calibration_status(instrument_ids)

    # Check safety approval (raises exception if not approved)
    await check_safety_approval(run_id, "ion_implant", HazardLevel.HIGH)

    # Start run
    # ... hardware control code ...

    # Create audit record
    create_audit_record(
        run_id=run_id,
        event_type="run_start",
        user=user,
        details={
            "target_dose": target_dose_cm2,
            "instrument_ids": [str(id) for id in instrument_ids]
        }
    )

    return {"status": "started", "run_id": str(run_id)}
```

## Best Practices

1. **Always Check Calibrations First:** Before requesting approval or starting a run
2. **Request Approvals Early:** HIGH/CRITICAL processes require dual approval which takes time
3. **Document Mitigations:** Clearly explain safety mitigations in approval requests
4. **Use E-Signatures:** All approvals and critical events should be cryptographically signed
5. **Review Audit Trail:** Regularly review audit records for compliance
6. **Monitor Expiring Calibrations:** Set up alerts for calibrations expiring within 30 days
7. **Update Uncertainty Budgets:** Recalculate uncertainties when instruments are recalibrated

## Testing

Access the interactive API documentation at: `http://localhost:8003/docs`

All endpoints can be tested through the Swagger UI with example payloads and responses.

## Future Enhancements

- Database persistence (currently uses in-memory storage)
- Real integration with LIMS for e-signatures
- Automated notifications for expiring calibrations
- Integration with equipment interlocks
- Role-based access control for approvals
- Calibration scheduling and reminders
- Statistical analysis of uncertainty trends
