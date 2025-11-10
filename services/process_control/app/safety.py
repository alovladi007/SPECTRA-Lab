"""Safety, Calibration, and Governance module for Process Control."""

from fastapi import HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class HazardLevel(str, Enum):
    """Hazard levels for safety classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CalibrationStatus(str, Enum):
    """Calibration status."""
    VALID = "VALID"
    EXPIRING_SOON = "EXPIRING_SOON"  # Within 30 days
    EXPIRED = "EXPIRED"
    NOT_CALIBRATED = "NOT_CALIBRATED"


class ApprovalStatus(str, Enum):
    """Approval status for safety gates."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class InstrumentType(str, Enum):
    """Types of instruments requiring calibration."""
    # Ion Implantation
    BEAM_CURRENT_INTEGRATOR = "beam_current_integrator"
    ANALYZER_MAGNET_PROBE = "analyzer_magnet_probe"
    VACUUM_GAUGE = "vacuum_gauge"

    # RTP
    PYROMETER = "pyrometer"
    THERMOCOUPLE = "thermocouple"
    PRESSURE_GAUGE = "pressure_gauge"
    MASS_FLOW_CONTROLLER = "mass_flow_controller"


# ============================================================================
# Schemas
# ============================================================================

class CalibrationRecord(BaseModel):
    """Calibration record for instruments."""
    id: UUID = Field(default_factory=uuid4)
    instrument_id: UUID
    instrument_type: InstrumentType
    instrument_name: str
    calibration_date: datetime
    expiry_date: datetime
    calibrated_by: str
    certificate_number: str
    uncertainty_pct: float
    standard_used: str
    next_cal_due: datetime
    notes: Optional[str] = None

    @property
    def status(self) -> CalibrationStatus:
        """Calculate calibration status."""
        now = datetime.now()
        if now > self.expiry_date:
            return CalibrationStatus.EXPIRED
        elif (self.expiry_date - now).days <= 30:
            return CalibrationStatus.EXPIRING_SOON
        return CalibrationStatus.VALID

    @property
    def days_until_expiry(self) -> int:
        """Days until calibration expires."""
        return (self.expiry_date - datetime.now()).days


class UncertaintyBudget(BaseModel):
    """Uncertainty budget for measurements."""
    parameter: str
    nominal_value: float
    unit: str
    type_a_uncertainty: float  # Statistical
    type_b_uncertainty: float  # Systematic
    combined_uncertainty: float
    expanded_uncertainty: float  # k=2, 95% confidence
    coverage_factor: float = 2.0
    contributors: Dict[str, float]  # Source -> contribution


class IonImplantUncertaintyBudget(BaseModel):
    """Uncertainty budget for ion implantation."""
    dose_uncertainty: UncertaintyBudget
    range_uncertainty: UncertaintyBudget
    straggle_uncertainty: UncertaintyBudget
    contributors: Dict[str, Dict[str, float]] = {
        "dose": {
            "current_integrator_accuracy": 0.0,
            "scan_uniformity": 0.0,
            "charge_collection": 0.0
        },
        "range": {
            "energy_uncertainty": 0.0,
            "species_purity": 0.0,
            "tilt_angle_error": 0.0
        },
        "straggle": {
            "energy_spread": 0.0,
            "target_composition": 0.0
        }
    }


class RTPUncertaintyBudget(BaseModel):
    """Uncertainty budget for RTP."""
    temperature_uncertainty: UncertaintyBudget
    ramp_rate_uncertainty: UncertaintyBudget
    contributors: Dict[str, Dict[str, float]] = {
        "temperature": {
            "emissivity_error": 0.0,
            "pyrometer_calibration": 0.0,
            "sensor_lag": 0.0,
            "spatial_nonuniformity": 0.0
        },
        "ramp_rate": {
            "setpoint_tracking": 0.0,
            "power_control": 0.0,
            "thermal_inertia": 0.0
        }
    }


class SafetyApproval(BaseModel):
    """Safety approval record."""
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    process_type: Literal["ion_implant", "rtp"]
    hazard_level: HazardLevel
    requested_by: str
    requested_at: datetime = Field(default_factory=datetime.now)
    approver_1: Optional[str] = None
    approved_at_1: Optional[datetime] = None
    approver_2: Optional[str] = None  # Required for HIGH/CRITICAL
    approved_at_2: Optional[datetime] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    rejection_reason: Optional[str] = None
    hazards: List[str] = []
    mitigations: List[str] = []

    @property
    def requires_dual_approval(self) -> bool:
        """Check if dual approval is required."""
        return self.hazard_level in [HazardLevel.HIGH, HazardLevel.CRITICAL]

    @property
    def is_fully_approved(self) -> bool:
        """Check if all required approvals are obtained."""
        if self.status != ApprovalStatus.APPROVED:
            return False
        if self.requires_dual_approval:
            return self.approver_1 is not None and self.approver_2 is not None
        return self.approver_1 is not None


class AuditRecord(BaseModel):
    """Immutable audit record."""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    run_id: UUID
    event_type: Literal["approval_request", "approval_granted", "approval_rejected",
                        "run_start", "run_complete", "run_abort", "e_signature"]
    user: str
    details: Dict[str, Any]
    e_signature: Optional[str] = None  # Cryptographic signature
    previous_record_hash: Optional[str] = None  # For blockchain-like integrity


class SOPGate(BaseModel):
    """Standard Operating Procedure gate definition."""
    id: UUID = Field(default_factory=uuid4)
    process_type: Literal["ion_implant", "rtp"]
    name: str
    description: str
    hazard_level: HazardLevel
    hazards: List[str]
    required_ppe: List[str]
    lockout_tagout_required: bool
    interlock_checks: List[str]
    emergency_procedures: List[str]
    required_calibrations: List[InstrumentType]


# ============================================================================
# Mock Data Store (Replace with database in production)
# ============================================================================

# Calibration records
_calibration_records: Dict[UUID, CalibrationRecord] = {}

# Safety approvals
_safety_approvals: Dict[UUID, SafetyApproval] = {}

# Audit trail
_audit_records: List[AuditRecord] = []

# SOP gates
_sop_gates: Dict[str, SOPGate] = {
    "ion_implant_standard": SOPGate(
        process_type="ion_implant",
        name="Standard Ion Implantation",
        description="Standard ion implantation procedure",
        hazard_level=HazardLevel.HIGH,
        hazards=[
            "High voltage (up to 500 kV)",
            "Vacuum hazard",
            "Ionizing radiation (X-rays)",
            "Toxic/flammable gases",
            "Electrical shock"
        ],
        required_ppe=["Safety glasses", "Lab coat", "Closed-toe shoes"],
        lockout_tagout_required=True,
        interlock_checks=[
            "Chamber door interlock",
            "Beam current interlock",
            "Vacuum level interlock",
            "Cooling water flow"
        ],
        emergency_procedures=[
            "Press EMERGENCY STOP button",
            "Evacuate area if gas leak detected",
            "Contact radiation safety officer if X-ray alarm"
        ],
        required_calibrations=[
            InstrumentType.BEAM_CURRENT_INTEGRATOR,
            InstrumentType.ANALYZER_MAGNET_PROBE,
            InstrumentType.VACUUM_GAUGE
        ]
    ),
    "rtp_high_temp": SOPGate(
        process_type="rtp",
        name="High Temperature RTP",
        description="RTP processes above 800°C",
        hazard_level=HazardLevel.HIGH,
        hazards=[
            "High temperature (up to 1200°C)",
            "Hot surfaces",
            "Thermal radiation",
            "Reactive gases",
            "Thermal shock hazard"
        ],
        required_ppe=["Heat-resistant gloves", "Safety glasses", "Lab coat"],
        lockout_tagout_required=True,
        interlock_checks=[
            "Chamber door interlock",
            "Temperature limit interlock",
            "Cooling water flow",
            "Gas flow interlocks"
        ],
        emergency_procedures=[
            "Press EMERGENCY STOP button",
            "Allow system to cool naturally",
            "Do not open chamber until temperature < 50°C"
        ],
        required_calibrations=[
            InstrumentType.PYROMETER,
            InstrumentType.THERMOCOUPLE,
            InstrumentType.PRESSURE_GAUGE,
            InstrumentType.MASS_FLOW_CONTROLLER
        ]
    )
}


# ============================================================================
# Safety Dependencies
# ============================================================================

async def check_calibration_status(
    instrument_ids: List[UUID]
) -> Dict[UUID, CalibrationRecord]:
    """
    Check calibration status for required instruments.
    Raises HTTPException if any calibration is expired.
    """
    calibrations = {}
    expired = []
    expiring_soon = []

    for inst_id in instrument_ids:
        if inst_id not in _calibration_records:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "CALIBRATION_NOT_FOUND",
                    "message": f"No calibration record found for instrument {inst_id}",
                    "instrument_id": str(inst_id)
                }
            )

        cal = _calibration_records[inst_id]
        calibrations[inst_id] = cal

        if cal.status == CalibrationStatus.EXPIRED:
            expired.append(cal)
        elif cal.status == CalibrationStatus.EXPIRING_SOON:
            expiring_soon.append(cal)

    if expired:
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "CALIBRATION_EXPIRED",
                "message": "Cannot proceed: One or more instruments have expired calibrations",
                "expired_instruments": [
                    {
                        "id": str(cal.id),
                        "name": cal.instrument_name,
                        "type": cal.instrument_type,
                        "expired_date": cal.expiry_date.isoformat(),
                        "days_overdue": -cal.days_until_expiry
                    }
                    for cal in expired
                ],
                "action_required": "Update instrument calibrations before proceeding"
            }
        )

    # Warning for expiring soon (but allow to proceed)
    if expiring_soon:
        # Could log warning or send notification
        pass

    return calibrations


async def check_safety_approval(
    run_id: UUID,
    process_type: Literal["ion_implant", "rtp"],
    hazard_level: Optional[HazardLevel] = None
) -> SafetyApproval:
    """
    Check if safety approval is obtained for the run.
    Raises HTTPException if approval is required but not granted.
    """
    # Find approval for this run
    approval = None
    for appr in _safety_approvals.values():
        if appr.run_id == run_id:
            approval = appr
            break

    # Determine required hazard level
    if hazard_level is None:
        # Infer from SOP gates
        gate_key = f"{process_type}_standard"
        if gate_key in _sop_gates:
            hazard_level = _sop_gates[gate_key].hazard_level
        else:
            hazard_level = HazardLevel.MEDIUM

    # Check if approval is required
    requires_approval = hazard_level in [HazardLevel.HIGH, HazardLevel.CRITICAL]

    if not requires_approval:
        return None

    if approval is None:
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "APPROVAL_REQUIRED",
                "message": f"Safety approval required for {hazard_level} hazard level process",
                "hazard_level": hazard_level,
                "process_type": process_type,
                "run_id": str(run_id),
                "action_required": "Submit safety approval request before proceeding"
            }
        )

    if approval.status == ApprovalStatus.REJECTED:
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "APPROVAL_REJECTED",
                "message": "Safety approval was rejected",
                "rejection_reason": approval.rejection_reason,
                "run_id": str(run_id)
            }
        )

    if approval.status == ApprovalStatus.PENDING:
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "APPROVAL_PENDING",
                "message": "Safety approval is pending",
                "requested_at": approval.requested_at.isoformat(),
                "approver_1": approval.approver_1,
                "approver_2": approval.approver_2,
                "requires_dual_approval": approval.requires_dual_approval,
                "run_id": str(run_id)
            }
        )

    if not approval.is_fully_approved:
        required_approvals = 2 if approval.requires_dual_approval else 1
        obtained_approvals = sum([
            approval.approver_1 is not None,
            approval.approver_2 is not None
        ])

        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "INSUFFICIENT_APPROVALS",
                "message": f"Insufficient approvals: {obtained_approvals}/{required_approvals}",
                "required_approvals": required_approvals,
                "obtained_approvals": obtained_approvals,
                "run_id": str(run_id)
            }
        )

    return approval


def create_audit_record(
    run_id: UUID,
    event_type: str,
    user: str,
    details: Dict[str, Any],
    e_signature: Optional[str] = None
) -> AuditRecord:
    """Create an immutable audit record."""
    # Get hash of previous record for chain integrity
    previous_hash = None
    if _audit_records:
        # In production, use actual cryptographic hash
        previous_hash = str(_audit_records[-1].id)

    record = AuditRecord(
        run_id=run_id,
        event_type=event_type,
        user=user,
        details=details,
        e_signature=e_signature,
        previous_record_hash=previous_hash
    )

    _audit_records.append(record)
    return record


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_ion_implant_uncertainty(
    energy_keV: float,
    dose_cm2: float,
    current_mA: float,
    calibration_uncertainty: float
) -> IonImplantUncertaintyBudget:
    """Calculate uncertainty budget for ion implantation."""
    # Dose uncertainty
    integrator_error = calibration_uncertainty  # From calibration
    uniformity_error = 0.02  # 2% typical
    collection_error = 0.01  # 1% typical

    dose_combined = (integrator_error**2 + uniformity_error**2 + collection_error**2)**0.5

    dose_unc = UncertaintyBudget(
        parameter="dose",
        nominal_value=dose_cm2,
        unit="ions/cm²",
        type_a_uncertainty=uniformity_error * dose_cm2,
        type_b_uncertainty=integrator_error * dose_cm2,
        combined_uncertainty=dose_combined * dose_cm2,
        expanded_uncertainty=2.0 * dose_combined * dose_cm2,
        contributors={
            "current_integrator": integrator_error,
            "scan_uniformity": uniformity_error,
            "charge_collection": collection_error
        }
    )

    # Range uncertainty (simplified model)
    energy_error = 0.01  # 1% energy uncertainty
    species_error = 0.005  # 0.5%

    range_nm = energy_keV * 0.3  # Simplified
    range_combined = (energy_error**2 + species_error**2)**0.5

    range_unc = UncertaintyBudget(
        parameter="projected_range",
        nominal_value=range_nm,
        unit="nm",
        type_a_uncertainty=0.0,
        type_b_uncertainty=range_combined * range_nm,
        combined_uncertainty=range_combined * range_nm,
        expanded_uncertainty=2.0 * range_combined * range_nm,
        contributors={
            "energy_uncertainty": energy_error,
            "species_purity": species_error
        }
    )

    # Straggle uncertainty
    straggle_nm = energy_keV * 0.1
    straggle_error = 0.15  # 15% typical for straggle

    straggle_unc = UncertaintyBudget(
        parameter="straggle",
        nominal_value=straggle_nm,
        unit="nm",
        type_a_uncertainty=0.0,
        type_b_uncertainty=straggle_error * straggle_nm,
        combined_uncertainty=straggle_error * straggle_nm,
        expanded_uncertainty=2.0 * straggle_error * straggle_nm,
        contributors={
            "energy_spread": 0.10,
            "target_composition": 0.05
        }
    )

    return IonImplantUncertaintyBudget(
        dose_uncertainty=dose_unc,
        range_uncertainty=range_unc,
        straggle_uncertainty=straggle_unc
    )


def calculate_rtp_uncertainty(
    temperature_C: float,
    emissivity: float,
    pyrometer_cal_uncertainty: float
) -> RTPUncertaintyBudget:
    """Calculate uncertainty budget for RTP."""
    # Temperature uncertainty
    emissivity_error = 0.05  # ±0.05 emissivity uncertainty
    temp_from_emissivity = temperature_C * emissivity_error / emissivity * 0.25  # Simplified

    pyrometer_error = pyrometer_cal_uncertainty
    sensor_lag_error = 2.0  # °C typical lag error
    spatial_error = 5.0  # °C typical spatial non-uniformity

    temp_combined = (temp_from_emissivity**2 + pyrometer_error**2 +
                     sensor_lag_error**2 + spatial_error**2)**0.5

    temp_unc = UncertaintyBudget(
        parameter="temperature",
        nominal_value=temperature_C,
        unit="°C",
        type_a_uncertainty=spatial_error,
        type_b_uncertainty=(temp_from_emissivity**2 + pyrometer_error**2 +
                           sensor_lag_error**2)**0.5,
        combined_uncertainty=temp_combined,
        expanded_uncertainty=2.0 * temp_combined,
        contributors={
            "emissivity_error": temp_from_emissivity,
            "pyrometer_calibration": pyrometer_error,
            "sensor_lag": sensor_lag_error,
            "spatial_nonuniformity": spatial_error
        }
    )

    # Ramp rate uncertainty
    ramp_error = 5.0  # °C/s typical

    ramp_unc = UncertaintyBudget(
        parameter="ramp_rate",
        nominal_value=50.0,  # Typical
        unit="°C/s",
        type_a_uncertainty=3.0,
        type_b_uncertainty=4.0,
        combined_uncertainty=ramp_error,
        expanded_uncertainty=2.0 * ramp_error,
        contributors={
            "setpoint_tracking": 3.0,
            "power_control": 2.0,
            "thermal_inertia": 2.0
        }
    )

    return RTPUncertaintyBudget(
        temperature_uncertainty=temp_unc,
        ramp_rate_uncertainty=ramp_unc
    )
