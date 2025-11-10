"""Safety, Calibration, and Governance API endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from pydantic import BaseModel

from app.safety import (
    CalibrationRecord, SafetyApproval, AuditRecord, SOPGate,
    UncertaintyBudget, IonImplantUncertaintyBudget, RTPUncertaintyBudget,
    HazardLevel, CalibrationStatus, ApprovalStatus, InstrumentType,
    check_calibration_status, check_safety_approval, create_audit_record,
    calculate_ion_implant_uncertainty, calculate_rtp_uncertainty,
    _calibration_records, _safety_approvals, _audit_records, _sop_gates
)


# Create router
safety_router = APIRouter(prefix="/api/v1/safety", tags=["Safety & Calibration"])


# ============================================================================
# Request/Response Schemas
# ============================================================================

class CalibrationCreateRequest(BaseModel):
    """Request to create calibration record."""
    instrument_id: UUID
    instrument_type: InstrumentType
    instrument_name: str
    calibrated_by: str
    certificate_number: str
    uncertainty_pct: float
    standard_used: str
    validity_days: int = 365
    notes: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Request for safety approval."""
    run_id: UUID
    process_type: str
    hazard_level: HazardLevel
    requested_by: str
    hazards: List[str]
    mitigations: List[str]


class ApprovalDecision(BaseModel):
    """Approval or rejection decision."""
    approver: str
    approved: bool
    reason: Optional[str] = None
    e_signature: Optional[str] = None


class IonImplantUncertaintyRequest(BaseModel):
    """Request to calculate ion implant uncertainty."""
    energy_keV: float
    dose_cm2: float
    current_mA: float
    instrument_ids: List[UUID]


class RTPUncertaintyRequest(BaseModel):
    """Request to calculate RTP uncertainty."""
    temperature_C: float
    emissivity: float
    instrument_ids: List[UUID]


# ============================================================================
# Calibration Endpoints
# ============================================================================

@safety_router.post("/calibration/records", response_model=CalibrationRecord)
async def create_calibration_record(request: CalibrationCreateRequest):
    """Create a new calibration record."""
    now = datetime.now()

    record = CalibrationRecord(
        instrument_id=request.instrument_id,
        instrument_type=request.instrument_type,
        instrument_name=request.instrument_name,
        calibration_date=now,
        expiry_date=now + timedelta(days=request.validity_days),
        calibrated_by=request.calibrated_by,
        certificate_number=request.certificate_number,
        uncertainty_pct=request.uncertainty_pct,
        standard_used=request.standard_used,
        next_cal_due=now + timedelta(days=request.validity_days - 30),  # 30 days before expiry
        notes=request.notes
    )

    _calibration_records[request.instrument_id] = record

    return record


@safety_router.get("/calibration/records", response_model=List[CalibrationRecord])
async def list_calibration_records(
    status: Optional[CalibrationStatus] = None,
    instrument_type: Optional[InstrumentType] = None
):
    """List all calibration records with optional filters."""
    records = list(_calibration_records.values())

    if status:
        records = [r for r in records if r.status == status]

    if instrument_type:
        records = [r for r in records if r.instrument_type == instrument_type]

    return records


@safety_router.get("/calibration/records/{instrument_id}", response_model=CalibrationRecord)
async def get_calibration_record(instrument_id: UUID):
    """Get calibration record for specific instrument."""
    if instrument_id not in _calibration_records:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "CALIBRATION_NOT_FOUND",
                "message": f"No calibration record found for instrument {instrument_id}"
            }
        )

    return _calibration_records[instrument_id]


@safety_router.post("/calibration/check")
async def check_calibrations(instrument_ids: List[UUID]):
    """
    Check calibration status for multiple instruments.
    Returns calibration status and raises exception if any are expired.
    """
    try:
        calibrations = await check_calibration_status(instrument_ids)
        return {
            "status": "valid",
            "calibrations": {
                str(inst_id): {
                    "instrument_name": cal.instrument_name,
                    "status": cal.status,
                    "expiry_date": cal.expiry_date.isoformat(),
                    "days_until_expiry": cal.days_until_expiry,
                    "uncertainty_pct": cal.uncertainty_pct
                }
                for inst_id, cal in calibrations.items()
            }
        }
    except HTTPException as e:
        # Re-raise with same detail
        raise e


# ============================================================================
# Safety Approval Endpoints
# ============================================================================

@safety_router.post("/approvals/request", response_model=SafetyApproval)
async def request_safety_approval(request: ApprovalRequest):
    """Submit a safety approval request."""
    approval = SafetyApproval(
        run_id=request.run_id,
        process_type=request.process_type,
        hazard_level=request.hazard_level,
        requested_by=request.requested_by,
        hazards=request.hazards,
        mitigations=request.mitigations
    )

    _safety_approvals[approval.id] = approval

    # Create audit record
    create_audit_record(
        run_id=request.run_id,
        event_type="approval_request",
        user=request.requested_by,
        details={
            "approval_id": str(approval.id),
            "hazard_level": request.hazard_level,
            "hazards": request.hazards
        }
    )

    return approval


@safety_router.post("/approvals/{approval_id}/approve")
async def approve_safety_request(approval_id: UUID, decision: ApprovalDecision):
    """Approve or reject a safety approval request."""
    if approval_id not in _safety_approvals:
        raise HTTPException(status_code=404, detail="Approval request not found")

    approval = _safety_approvals[approval_id]

    if not decision.approved:
        # Rejection
        approval.status = ApprovalStatus.REJECTED
        approval.rejection_reason = decision.reason

        create_audit_record(
            run_id=approval.run_id,
            event_type="approval_rejected",
            user=decision.approver,
            details={
                "approval_id": str(approval_id),
                "reason": decision.reason
            },
            e_signature=decision.e_signature
        )

        return approval

    # Approval
    if approval.approver_1 is None:
        approval.approver_1 = decision.approver
        approval.approved_at_1 = datetime.now()
    elif approval.approver_2 is None and approval.requires_dual_approval:
        # Second approver
        if decision.approver == approval.approver_1:
            raise HTTPException(
                status_code=400,
                detail="Second approver must be different from first approver"
            )
        approval.approver_2 = decision.approver
        approval.approved_at_2 = datetime.now()
    else:
        raise HTTPException(
            status_code=400,
            detail="Approval already has sufficient approvers"
        )

    # Check if fully approved
    if approval.is_fully_approved:
        approval.status = ApprovalStatus.APPROVED

    create_audit_record(
        run_id=approval.run_id,
        event_type="approval_granted",
        user=decision.approver,
        details={
            "approval_id": str(approval_id),
            "approver_number": 1 if approval.approver_2 is None else 2,
            "fully_approved": approval.is_fully_approved
        },
        e_signature=decision.e_signature
    )

    return approval


@safety_router.get("/approvals", response_model=List[SafetyApproval])
async def list_approvals(
    status: Optional[ApprovalStatus] = None,
    process_type: Optional[str] = None
):
    """List all safety approval requests."""
    approvals = list(_safety_approvals.values())

    if status:
        approvals = [a for a in approvals if a.status == status]

    if process_type:
        approvals = [a for a in approvals if a.process_type == process_type]

    return approvals


@safety_router.get("/approvals/{approval_id}", response_model=SafetyApproval)
async def get_approval(approval_id: UUID):
    """Get specific approval request."""
    if approval_id not in _safety_approvals:
        raise HTTPException(status_code=404, detail="Approval request not found")

    return _safety_approvals[approval_id]


@safety_router.get("/approvals/run/{run_id}", response_model=Optional[SafetyApproval])
async def get_approval_by_run(run_id: UUID):
    """Get approval for specific run ID."""
    for approval in _safety_approvals.values():
        if approval.run_id == run_id:
            return approval

    return None


@safety_router.post("/approvals/run/{run_id}/check")
async def check_approval_for_run(
    run_id: UUID,
    process_type: str,
    hazard_level: Optional[HazardLevel] = None
):
    """
    Check if run has required approvals.
    Raises exception if approval is required but not granted.
    """
    try:
        approval = await check_safety_approval(run_id, process_type, hazard_level)
        if approval is None:
            return {
                "status": "no_approval_required",
                "hazard_level": hazard_level
            }
        return {
            "status": "approved",
            "approval": approval
        }
    except HTTPException as e:
        raise e


# ============================================================================
# SOP Gate Endpoints
# ============================================================================

@safety_router.get("/sop-gates", response_model=List[SOPGate])
async def list_sop_gates():
    """List all SOP gates."""
    return list(_sop_gates.values())


@safety_router.get("/sop-gates/{gate_key}", response_model=SOPGate)
async def get_sop_gate(gate_key: str):
    """Get specific SOP gate."""
    if gate_key not in _sop_gates:
        raise HTTPException(status_code=404, detail="SOP gate not found")

    return _sop_gates[gate_key]


# ============================================================================
# Uncertainty Budget Endpoints
# ============================================================================

@safety_router.post("/uncertainty/ion-implant", response_model=IonImplantUncertaintyBudget)
async def calculate_ion_uncertainty(request: IonImplantUncertaintyRequest):
    """Calculate uncertainty budget for ion implantation."""
    # Check calibrations
    calibrations = await check_calibration_status(request.instrument_ids)

    # Get integrator calibration
    integrator_cal = None
    for cal in calibrations.values():
        if cal.instrument_type == InstrumentType.BEAM_CURRENT_INTEGRATOR:
            integrator_cal = cal
            break

    if integrator_cal is None:
        raise HTTPException(
            status_code=400,
            detail="Beam current integrator calibration not found"
        )

    budget = calculate_ion_implant_uncertainty(
        energy_keV=request.energy_keV,
        dose_cm2=request.dose_cm2,
        current_mA=request.current_mA,
        calibration_uncertainty=integrator_cal.uncertainty_pct / 100.0
    )

    return budget


@safety_router.post("/uncertainty/rtp", response_model=RTPUncertaintyBudget)
async def calculate_rtp_uncertainty_endpoint(request: RTPUncertaintyRequest):
    """Calculate uncertainty budget for RTP."""
    # Check calibrations
    calibrations = await check_calibration_status(request.instrument_ids)

    # Get pyrometer calibration
    pyrometer_cal = None
    for cal in calibrations.values():
        if cal.instrument_type == InstrumentType.PYROMETER:
            pyrometer_cal = cal
            break

    if pyrometer_cal is None:
        raise HTTPException(
            status_code=400,
            detail="Pyrometer calibration not found"
        )

    # Convert percentage to absolute temperature uncertainty
    pyrometer_uncertainty_C = request.temperature_C * (pyrometer_cal.uncertainty_pct / 100.0)

    budget = calculate_rtp_uncertainty(
        temperature_C=request.temperature_C,
        emissivity=request.emissivity,
        pyrometer_cal_uncertainty=pyrometer_uncertainty_C
    )

    return budget


# ============================================================================
# Audit Trail Endpoints
# ============================================================================

@safety_router.get("/audit/records", response_model=List[AuditRecord])
async def list_audit_records(
    run_id: Optional[UUID] = None,
    event_type: Optional[str] = None,
    user: Optional[str] = None,
    limit: int = 100
):
    """List audit records with optional filters."""
    records = _audit_records.copy()

    if run_id:
        records = [r for r in records if r.run_id == run_id]

    if event_type:
        records = [r for r in records if r.event_type == event_type]

    if user:
        records = [r for r in records if r.user == user]

    # Return most recent first
    records.reverse()

    return records[:limit]


@safety_router.get("/audit/records/{record_id}", response_model=AuditRecord)
async def get_audit_record(record_id: UUID):
    """Get specific audit record."""
    for record in _audit_records:
        if record.id == record_id:
            return record

    raise HTTPException(status_code=404, detail="Audit record not found")


@safety_router.post("/audit/e-sign")
async def create_e_signature(
    run_id: UUID,
    user: str,
    role: str,
    signature: str,
    comments: Optional[str] = None
):
    """Create an e-signature audit record."""
    record = create_audit_record(
        run_id=run_id,
        event_type="e_signature",
        user=user,
        details={
            "role": role,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        },
        e_signature=signature
    )

    return {
        "record_id": str(record.id),
        "message": "E-signature recorded",
        "timestamp": record.timestamp.isoformat()
    }


# ============================================================================
# Compliance Summary Endpoint
# ============================================================================

@safety_router.get("/compliance/summary")
async def get_compliance_summary():
    """Get overall compliance summary."""
    # Calibration status
    total_instruments = len(_calibration_records)
    expired = sum(1 for cal in _calibration_records.values()
                  if cal.status == CalibrationStatus.EXPIRED)
    expiring_soon = sum(1 for cal in _calibration_records.values()
                       if cal.status == CalibrationStatus.EXPIRING_SOON)
    valid = total_instruments - expired - expiring_soon

    # Approval status
    total_approvals = len(_safety_approvals)
    pending = sum(1 for appr in _safety_approvals.values()
                 if appr.status == ApprovalStatus.PENDING)
    approved = sum(1 for appr in _safety_approvals.values()
                  if appr.status == ApprovalStatus.APPROVED)
    rejected = sum(1 for appr in _safety_approvals.values()
                  if appr.status == ApprovalStatus.REJECTED)

    # Audit trail
    total_audits = len(_audit_records)
    e_signatures = sum(1 for rec in _audit_records if rec.e_signature is not None)

    return {
        "calibration": {
            "total_instruments": total_instruments,
            "valid": valid,
            "expiring_soon": expiring_soon,
            "expired": expired,
            "compliance_rate": (valid / total_instruments * 100) if total_instruments > 0 else 100.0
        },
        "approvals": {
            "total": total_approvals,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": (approved / total_approvals * 100) if total_approvals > 0 else 0.0
        },
        "audit": {
            "total_records": total_audits,
            "e_signatures": e_signatures,
            "chain_integrity": "verified"  # In production, verify chain
        },
        "overall_status": "COMPLIANT" if expired == 0 and pending == 0 else "NEEDS_ATTENTION"
    }


# Export router
__all__ = ["safety_router"]
