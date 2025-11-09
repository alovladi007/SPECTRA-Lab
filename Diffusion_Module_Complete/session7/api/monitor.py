"""
SPC Monitoring API Endpoint - Session 7

FastAPI endpoint for real-time SPC monitoring of semiconductor process KPIs.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import numpy as np

# These would be imported from FastAPI in production
# from fastapi import APIRouter, HTTPException
# router = APIRouter(prefix="/spc", tags=["SPC Monitoring"])


class KPIType(str, Enum):
    """Types of process KPIs."""
    JUNCTION_DEPTH = "junction_depth"
    SHEET_RESISTANCE = "sheet_resistance"
    OXIDE_THICKNESS = "oxide_thickness"
    UNIFORMITY = "uniformity"


class MonitorRequest(BaseModel):
    """Request for SPC monitoring."""
    kpi_type: KPIType
    data: List[float] = Field(..., min_items=10)
    timestamps: Optional[List[datetime]] = None
    target: Optional[float] = None
    sigma: Optional[float] = None
    enable_rules: bool = True
    enable_ewma: bool = True
    enable_cusum: bool = True
    enable_bocpd: bool = True
    ewma_lambda: float = Field(0.2, ge=0.0, le=1.0)
    cusum_shift_size: float = Field(1.0, gt=0.0)
    bocpd_hazard_lambda: float = Field(250.0, gt=0.0)


class RuleViolationResponse(BaseModel):
    """Rule violation in response."""
    rule: str
    index: int
    timestamp: Optional[datetime]
    severity: str
    description: str
    metric_value: float


class EWMAViolationResponse(BaseModel):
    """EWMA violation in response."""
    index: int
    timestamp: Optional[datetime]
    ewma_value: float
    data_value: float
    limit_exceeded: str
    description: str


class CUSUMViolationResponse(BaseModel):
    """CUSUM violation in response."""
    index: int
    timestamp: Optional[datetime]
    cusum_high: float
    cusum_low: float
    data_value: float
    direction: str
    description: str


class ChangePointResponse(BaseModel):
    """Change point in response."""
    index: int
    timestamp: Optional[datetime]
    run_length: int
    probability: float
    data_value: float
    description: str


class MonitorResponse(BaseModel):
    """Response from SPC monitoring."""
    kpi_type: KPIType
    n_samples: int
    rule_violations: List[RuleViolationResponse]
    ewma_violations: List[EWMAViolationResponse]
    cusum_violations: List[CUSUMViolationResponse]
    change_points: List[ChangePointResponse]
    summary: Dict[str, Any]
    status: str  # "IN_CONTROL", "WARNING", "OUT_OF_CONTROL"


def monitor_kpi(request: MonitorRequest) -> MonitorResponse:
    """
    Monitor KPI series for violations and change points.

    This function would be wrapped in a FastAPI endpoint:
    @router.post("/monitor")
    async def monitor_endpoint(request: MonitorRequest) -> MonitorResponse:
        return monitor_kpi(request)
    """
    from session7.spc import (
        quick_rule_check,
        quick_ewma_check,
        quick_cusum_check,
        quick_bocpd_check,
    )

    data = np.array(request.data)
    rule_violations_list = []
    ewma_violations_list = []
    cusum_violations_list = []
    change_points_list = []

    # Check Western Electric & Nelson rules
    if request.enable_rules:
        violations = quick_rule_check(
            data,
            centerline=request.target,
            sigma=request.sigma,
            timestamps=request.timestamps
        )

        for v in violations:
            rule_violations_list.append(RuleViolationResponse(
                rule=v.rule.value,
                index=v.index,
                timestamp=v.timestamp,
                severity=v.severity.value,
                description=v.description,
                metric_value=v.metric_value
            ))

    # Check EWMA
    if request.enable_ewma:
        _, ewma_violations = quick_ewma_check(
            data,
            lambda_=request.ewma_lambda,
            target=request.target,
            sigma=request.sigma,
            timestamps=request.timestamps
        )

        for v in ewma_violations:
            ewma_violations_list.append(EWMAViolationResponse(
                index=v.index,
                timestamp=v.timestamp,
                ewma_value=v.ewma_value,
                data_value=v.data_value,
                limit_exceeded=v.limit_exceeded,
                description=v.description
            ))

    # Check CUSUM
    if request.enable_cusum:
        _, _, cusum_violations = quick_cusum_check(
            data,
            target=request.target,
            sigma=request.sigma,
            shift_size=request.cusum_shift_size,
            timestamps=request.timestamps
        )

        for v in cusum_violations:
            cusum_violations_list.append(CUSUMViolationResponse(
                index=v.index,
                timestamp=v.timestamp,
                cusum_high=v.cusum_high,
                cusum_low=v.cusum_low,
                data_value=v.data_value,
                direction=v.direction,
                description=v.description
            ))

    # Check for change points
    if request.enable_bocpd:
        change_points, _ = quick_bocpd_check(
            data,
            hazard_lambda=request.bocpd_hazard_lambda,
            timestamps=request.timestamps
        )

        for cp in change_points:
            change_points_list.append(ChangePointResponse(
                index=cp.index,
                timestamp=cp.timestamp,
                run_length=cp.run_length,
                probability=cp.probability,
                data_value=cp.data_value,
                description=cp.description
            ))

    # Determine overall status
    critical_count = sum(1 for v in rule_violations_list if v.severity == "CRITICAL")
    warning_count = sum(1 for v in rule_violations_list if v.severity == "WARNING")
    total_violations = len(rule_violations_list) + len(ewma_violations_list) + len(cusum_violations_list)

    if critical_count > 0 or total_violations >= 3:
        status = "OUT_OF_CONTROL"
    elif warning_count > 0 or total_violations > 0:
        status = "WARNING"
    else:
        status = "IN_CONTROL"

    # Summary statistics
    summary = {
        "mean": float(np.mean(data)),
        "std_dev": float(np.std(data, ddof=1)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "range": float(np.ptp(data)),
        "total_violations": total_violations,
        "critical_violations": critical_count,
        "warning_violations": warning_count,
        "n_change_points": len(change_points_list),
    }

    return MonitorResponse(
        kpi_type=request.kpi_type,
        n_samples=len(data),
        rule_violations=rule_violations_list,
        ewma_violations=ewma_violations_list,
        cusum_violations=cusum_violations_list,
        change_points=change_points_list,
        summary=summary,
        status=status
    )
