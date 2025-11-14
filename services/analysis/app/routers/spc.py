"""
Centralized SPC (Statistical Process Control) Router
Aggregates SPC data from all manufacturing processes: Diffusion, CVD, Oxidation, Ion, RTP
Provides unified endpoints for cross-process SPC monitoring and analysis
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict, Field

import sys
from pathlib import Path

# Add services/shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from db.deps import get_db

# Import SPC models from different processes
from ..models.diffusion import (
    DiffusionSPCSeries,
    DiffusionSPCPoint,
)
from ..models.cvd import (
    CVDSPCSeries,
    CVDSPCPoint,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/spc", tags=["SPC - Statistical Process Control"])


# ============================================================================
# Unified Response Schemas
# ============================================================================

class UnifiedSPCSeriesSchema(BaseModel):
    """Unified SPC Series schema for all process types"""
    id: UUID
    org_id: UUID
    name: str
    parameter: str
    process_type: str = Field(
        ...,
        description="Process type: diffusion, cvd, oxidation, ion, rtp"
    )

    # Control limits
    chart_type: Optional[str] = Field(
        default="I_MR",
        description="Chart type: XBAR_R, I_MR, P, C, etc."
    )
    control_status: str = Field(
        default="IN_CONTROL",
        description="Control status: IN_CONTROL, OUT_OF_CONTROL, WARNING"
    )
    ucl: Optional[float] = Field(None, description="Upper Control Limit")
    lcl: Optional[float] = Field(None, description="Lower Control Limit")
    target: Optional[float] = Field(None, description="Target value")

    # Spec limits
    usl: Optional[float] = Field(None, description="Upper Specification Limit")
    lsl: Optional[float] = Field(None, description="Lower Specification Limit")

    # Statistics
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    cp: Optional[float] = Field(None, description="Process capability")
    cpk: Optional[float] = Field(None, description="Process capability index")

    # Counts
    sample_count: int = Field(default=0, description="Total number of samples")
    violation_count: int = Field(default=0, description="Number of violations")

    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class UnifiedSPCPointSchema(BaseModel):
    """Unified SPC Point schema for all process types"""
    id: UUID
    series_id: UUID
    value: float
    ts: datetime
    process_type: str = Field(
        ...,
        description="Process type: diffusion, cvd, oxidation, ion, rtp"
    )

    violates_rule: Optional[bool] = Field(None, description="Whether point violates control rules")
    rule_violations: Optional[List[str]] = Field(
        None,
        description="List of violated rule names"
    )

    # Optional run reference
    run_id: Optional[UUID] = Field(None, description="Associated process run ID")

    model_config = ConfigDict(from_attributes=True)


class SPCViolationSummary(BaseModel):
    """Summary of SPC violations"""
    series_id: UUID
    series_name: str
    parameter: str
    process_type: str
    violation_count: int
    latest_violation_time: Optional[datetime]
    control_status: str


class SPCDashboardResponse(BaseModel):
    """Dashboard summary of all SPC data"""
    total_series: int
    by_process: Dict[str, int]
    in_control_count: int
    out_of_control_count: int
    warning_count: int
    recent_violations: List[SPCViolationSummary]
    top_violating_parameters: List[Dict[str, Any]]
    capability_summary: Dict[str, Any]


# ============================================================================
# Helper Functions
# ============================================================================

def determine_control_status(series: Any, violation_count: int) -> str:
    """
    Determine control status based on series statistics and violations

    Args:
        series: SPC series object
        violation_count: Number of violations

    Returns:
        str: Control status (IN_CONTROL, OUT_OF_CONTROL, WARNING)
    """
    if violation_count == 0:
        return "IN_CONTROL"
    elif violation_count >= 3:
        return "OUT_OF_CONTROL"
    else:
        return "WARNING"


async def get_diffusion_series_with_counts(
    db: Session,
    org_id: UUID,
    status_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get diffusion SPC series with violation counts"""
    query = select(DiffusionSPCSeries).where(DiffusionSPCSeries.org_id == org_id)

    if status_filter and status_filter != "ALL":
        query = query.where(DiffusionSPCSeries.is_active == (status_filter == "IN_CONTROL"))

    result = db.execute(query)
    series_list = result.scalars().all()

    unified_series = []
    for series in series_list:
        # Count total points
        point_count_query = select(func.count()).where(DiffusionSPCPoint.series_id == series.id)
        sample_count = db.execute(point_count_query).scalar() or 0

        # Count violations
        violation_count_query = select(func.count()).where(
            and_(
                DiffusionSPCPoint.series_id == series.id,
                DiffusionSPCPoint.violation == True
            )
        )
        violation_count = db.execute(violation_count_query).scalar() or 0

        control_status = determine_control_status(series, violation_count)

        unified_series.append({
            "series": series,
            "sample_count": sample_count,
            "violation_count": violation_count,
            "control_status": control_status,
            "process_type": "diffusion"
        })

    return unified_series


async def get_cvd_series_with_counts(
    db: Session,
    org_id: UUID,
    status_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get CVD SPC series with violation counts"""
    query = select(CVDSPCSeries).where(CVDSPCSeries.org_id == org_id)

    result = db.execute(query)
    series_list = result.scalars().all()

    unified_series = []
    for series in series_list:
        # Count total points
        point_count_query = select(func.count()).where(CVDSPCPoint.series_id == series.id)
        sample_count = db.execute(point_count_query).scalar() or 0

        # Count violations
        violation_count_query = select(func.count()).where(
            and_(
                CVDSPCPoint.series_id == series.id,
                CVDSPCPoint.violation == True
            )
        )
        violation_count = db.execute(violation_count_query).scalar() or 0

        control_status = determine_control_status(series, violation_count)

        unified_series.append({
            "series": series,
            "sample_count": sample_count,
            "violation_count": violation_count,
            "control_status": control_status,
            "process_type": "cvd"
        })

    return unified_series


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/series", response_model=List[UnifiedSPCSeriesSchema])
async def get_all_spc_series(
    org_id: UUID = Query(..., description="Organization ID"),
    process_type: Optional[str] = Query(
        None,
        description="Filter by process type: diffusion, cvd, oxidation, ion, rtp"
    ),
    status: Optional[str] = Query(
        None,
        description="Filter by control status: IN_CONTROL, OUT_OF_CONTROL, WARNING, ALL"
    ),
    parameter: Optional[str] = Query(
        None,
        description="Filter by parameter name"
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Get all SPC series across all manufacturing processes

    Returns unified list of SPC series from Diffusion, CVD, Oxidation, Ion, and RTP processes
    with control status, violation counts, and process capability metrics.
    """
    try:
        all_series = []

        # Fetch diffusion series if not filtered by process_type or specifically requested
        if not process_type or process_type == "diffusion":
            diffusion_series = await get_diffusion_series_with_counts(db, org_id, status)
            all_series.extend(diffusion_series)

        # Fetch CVD series if not filtered by process_type or specifically requested
        if not process_type or process_type == "cvd":
            cvd_series = await get_cvd_series_with_counts(db, org_id, status)
            all_series.extend(cvd_series)

        # TODO: Add oxidation, ion, rtp when SPC models are available
        # if not process_type or process_type == "oxidation":
        #     oxidation_series = await get_oxidation_series_with_counts(db, org_id, status)
        #     all_series.extend(oxidation_series)

        # Filter by status if provided
        if status and status != "ALL":
            all_series = [s for s in all_series if s["control_status"] == status]

        # Filter by parameter if provided
        if parameter:
            all_series = [s for s in all_series if s["series"].parameter == parameter]

        # Apply pagination
        paginated_series = all_series[skip:skip + limit]

        # Convert to unified schema
        response = []
        for item in paginated_series:
            series = item["series"]
            response.append(UnifiedSPCSeriesSchema(
                id=series.id,
                org_id=series.org_id,
                name=series.name,
                parameter=series.parameter,
                process_type=item["process_type"],
                chart_type="I_MR",  # Default, could be extended
                control_status=item["control_status"],
                ucl=series.ucl,
                lcl=series.lcl,
                target=series.target,
                usl=series.usl,
                lsl=series.lsl,
                mean=series.mean,
                std_dev=series.std_dev,
                cp=series.cp,
                cpk=series.cpk,
                sample_count=item["sample_count"],
                violation_count=item["violation_count"],
                created_at=series.created_at,
                updated_at=series.updated_at
            ))

        logger.info(f"Retrieved {len(response)} SPC series for org {org_id}")
        return response

    except Exception as e:
        logger.exception(f"Error getting SPC series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/series/{series_id}/points", response_model=List[UnifiedSPCPointSchema])
async def get_spc_points(
    series_id: UUID,
    limit: int = Query(100, ge=1, le=10000, description="Maximum number of points to return"),
    skip: int = Query(0, ge=0),
    start_date: Optional[datetime] = Query(None, description="Filter points after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter points before this date"),
    db: Session = Depends(get_db),
):
    """
    Get SPC points for a specific series

    Automatically determines which table to query based on series_id and returns
    the data points with violation information.
    """
    try:
        # Try to find the series in diffusion tables
        diffusion_series_query = select(DiffusionSPCSeries).where(DiffusionSPCSeries.id == series_id)
        diffusion_series = db.execute(diffusion_series_query).scalar_one_or_none()

        if diffusion_series:
            # Query diffusion SPC points
            query = select(DiffusionSPCPoint).where(DiffusionSPCPoint.series_id == series_id)

            if start_date:
                query = query.where(DiffusionSPCPoint.ts >= start_date)
            if end_date:
                query = query.where(DiffusionSPCPoint.ts <= end_date)

            query = query.order_by(DiffusionSPCPoint.ts.desc()).offset(skip).limit(limit)
            result = db.execute(query)
            points = result.scalars().all()

            return [
                UnifiedSPCPointSchema(
                    id=p.id,
                    series_id=p.series_id,
                    value=p.value,
                    ts=p.ts,
                    process_type="diffusion",
                    violates_rule=p.violation,
                    rule_violations=p.violation_rules if p.violation_rules else None,
                    run_id=p.run_id
                )
                for p in points
            ]

        # Try CVD tables
        cvd_series_query = select(CVDSPCSeries).where(CVDSPCSeries.id == series_id)
        cvd_series = db.execute(cvd_series_query).scalar_one_or_none()

        if cvd_series:
            # Query CVD SPC points
            query = select(CVDSPCPoint).where(CVDSPCPoint.series_id == series_id)

            if start_date:
                query = query.where(CVDSPCPoint.ts >= start_date)
            if end_date:
                query = query.where(CVDSPCPoint.ts <= end_date)

            query = query.order_by(CVDSPCPoint.ts.desc()).offset(skip).limit(limit)
            result = db.execute(query)
            points = result.scalars().all()

            return [
                UnifiedSPCPointSchema(
                    id=p.id,
                    series_id=p.series_id,
                    value=p.value,
                    ts=p.ts,
                    process_type="cvd",
                    violates_rule=p.violation,
                    rule_violations=p.violation_rules if p.violation_rules else None,
                    run_id=p.cvd_run_id
                )
                for p in points
            ]

        # Series not found in any table
        raise HTTPException(
            status_code=404,
            detail=f"SPC series {series_id} not found in any process"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting SPC points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/violations", response_model=List[SPCViolationSummary])
async def get_spc_violations(
    org_id: UUID = Query(..., description="Organization ID"),
    start_date: Optional[datetime] = Query(
        None,
        description="Filter violations after this date"
    ),
    end_date: Optional[datetime] = Query(
        None,
        description="Filter violations before this date"
    ),
    process_type: Optional[str] = Query(
        None,
        description="Filter by process type"
    ),
    min_violations: int = Query(
        1,
        ge=1,
        description="Minimum number of violations to include"
    ),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Get all SPC violations across all processes

    Returns aggregated violation data for OUT_OF_CONTROL and WARNING series,
    sorted by violation count (descending).
    """
    try:
        violations = []

        # Get diffusion violations
        if not process_type or process_type == "diffusion":
            diffusion_series = await get_diffusion_series_with_counts(db, org_id)

            for item in diffusion_series:
                series = item["series"]
                violation_count = item["violation_count"]

                if violation_count >= min_violations:
                    # Get latest violation time
                    latest_violation_query = (
                        select(DiffusionSPCPoint.ts)
                        .where(
                            and_(
                                DiffusionSPCPoint.series_id == series.id,
                                DiffusionSPCPoint.violation == True
                            )
                        )
                        .order_by(DiffusionSPCPoint.ts.desc())
                        .limit(1)
                    )

                    if start_date:
                        latest_violation_query = latest_violation_query.where(
                            DiffusionSPCPoint.ts >= start_date
                        )
                    if end_date:
                        latest_violation_query = latest_violation_query.where(
                            DiffusionSPCPoint.ts <= end_date
                        )

                    latest_time = db.execute(latest_violation_query).scalar_one_or_none()

                    violations.append(SPCViolationSummary(
                        series_id=series.id,
                        series_name=series.name,
                        parameter=series.parameter,
                        process_type="diffusion",
                        violation_count=violation_count,
                        latest_violation_time=latest_time,
                        control_status=item["control_status"]
                    ))

        # Get CVD violations
        if not process_type or process_type == "cvd":
            cvd_series = await get_cvd_series_with_counts(db, org_id)

            for item in cvd_series:
                series = item["series"]
                violation_count = item["violation_count"]

                if violation_count >= min_violations:
                    # Get latest violation time
                    latest_violation_query = (
                        select(CVDSPCPoint.ts)
                        .where(
                            and_(
                                CVDSPCPoint.series_id == series.id,
                                CVDSPCPoint.violation == True
                            )
                        )
                        .order_by(CVDSPCPoint.ts.desc())
                        .limit(1)
                    )

                    if start_date:
                        latest_violation_query = latest_violation_query.where(
                            CVDSPCPoint.ts >= start_date
                        )
                    if end_date:
                        latest_violation_query = latest_violation_query.where(
                            CVDSPCPoint.ts <= end_date
                        )

                    latest_time = db.execute(latest_violation_query).scalar_one_or_none()

                    violations.append(SPCViolationSummary(
                        series_id=series.id,
                        series_name=series.name,
                        parameter=series.parameter,
                        process_type="cvd",
                        violation_count=violation_count,
                        latest_violation_time=latest_time,
                        control_status=item["control_status"]
                    ))

        # Sort by violation count descending
        violations.sort(key=lambda x: x.violation_count, reverse=True)

        # Apply limit
        violations = violations[:limit]

        logger.info(f"Retrieved {len(violations)} SPC violations for org {org_id}")
        return violations

    except Exception as e:
        logger.exception(f"Error getting SPC violations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard", response_model=SPCDashboardResponse)
async def get_spc_dashboard(
    org_id: UUID = Query(..., description="Organization ID"),
    db: Session = Depends(get_db),
):
    """
    Get SPC dashboard summary

    Returns comprehensive statistics including:
    - Total series count by process
    - In control / out of control counts
    - Recent violations
    - Top violating parameters
    - Process capability summary
    """
    try:
        # Get all series data
        diffusion_series = await get_diffusion_series_with_counts(db, org_id)
        cvd_series = await get_cvd_series_with_counts(db, org_id)

        all_series = diffusion_series + cvd_series

        # Calculate counts
        total_series = len(all_series)
        by_process = {
            "diffusion": len(diffusion_series),
            "cvd": len(cvd_series),
            "oxidation": 0,  # TODO: Add when available
            "ion": 0,
            "rtp": 0
        }

        in_control_count = len([s for s in all_series if s["control_status"] == "IN_CONTROL"])
        out_of_control_count = len([s for s in all_series if s["control_status"] == "OUT_OF_CONTROL"])
        warning_count = len([s for s in all_series if s["control_status"] == "WARNING"])

        # Get recent violations (last 10)
        all_violations = []
        for item in all_series:
            if item["violation_count"] > 0:
                series = item["series"]

                # Determine which point table to query
                if item["process_type"] == "diffusion":
                    latest_violation_query = (
                        select(DiffusionSPCPoint.ts)
                        .where(
                            and_(
                                DiffusionSPCPoint.series_id == series.id,
                                DiffusionSPCPoint.violation == True
                            )
                        )
                        .order_by(DiffusionSPCPoint.ts.desc())
                        .limit(1)
                    )
                else:  # cvd
                    latest_violation_query = (
                        select(CVDSPCPoint.ts)
                        .where(
                            and_(
                                CVDSPCPoint.series_id == series.id,
                                CVDSPCPoint.violation == True
                            )
                        )
                        .order_by(CVDSPCPoint.ts.desc())
                        .limit(1)
                    )

                latest_time = db.execute(latest_violation_query).scalar_one_or_none()

                all_violations.append(SPCViolationSummary(
                    series_id=series.id,
                    series_name=series.name,
                    parameter=series.parameter,
                    process_type=item["process_type"],
                    violation_count=item["violation_count"],
                    latest_violation_time=latest_time,
                    control_status=item["control_status"]
                ))

        # Sort by latest violation time and take top 10
        recent_violations = sorted(
            [v for v in all_violations if v.latest_violation_time],
            key=lambda x: x.latest_violation_time,
            reverse=True
        )[:10]

        # Get top violating parameters (by total violation count)
        parameter_violations = {}
        for item in all_series:
            param = item["series"].parameter
            if param not in parameter_violations:
                parameter_violations[param] = {
                    "parameter": param,
                    "total_violations": 0,
                    "series_count": 0
                }
            parameter_violations[param]["total_violations"] += item["violation_count"]
            parameter_violations[param]["series_count"] += 1

        top_violating_parameters = sorted(
            parameter_violations.values(),
            key=lambda x: x["total_violations"],
            reverse=True
        )[:10]

        # Calculate capability summary
        cpk_values = [s["series"].cpk for s in all_series if s["series"].cpk is not None]
        capability_summary = {
            "average_cpk": sum(cpk_values) / len(cpk_values) if cpk_values else None,
            "min_cpk": min(cpk_values) if cpk_values else None,
            "max_cpk": max(cpk_values) if cpk_values else None,
            "capable_count": len([cpk for cpk in cpk_values if cpk >= 1.33]),
            "marginal_count": len([cpk for cpk in cpk_values if 1.0 <= cpk < 1.33]),
            "incapable_count": len([cpk for cpk in cpk_values if cpk < 1.0])
        }

        logger.info(f"Generated SPC dashboard for org {org_id}")

        return SPCDashboardResponse(
            total_series=total_series,
            by_process=by_process,
            in_control_count=in_control_count,
            out_of_control_count=out_of_control_count,
            warning_count=warning_count,
            recent_violations=recent_violations,
            top_violating_parameters=top_violating_parameters,
            capability_summary=capability_summary
        )

    except Exception as e:
        logger.exception(f"Error generating SPC dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for SPC service"""
    return {
        "status": "healthy",
        "service": "spc-aggregation",
        "supported_processes": ["diffusion", "cvd"],
        "future_processes": ["oxidation", "ion", "rtp"],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/parameters")
async def get_available_parameters(
    org_id: UUID = Query(..., description="Organization ID"),
    process_type: Optional[str] = Query(None, description="Filter by process type"),
    db: Session = Depends(get_db),
):
    """
    Get list of all parameters being tracked across all processes

    Returns unique parameter names with their associated process types and series counts.
    """
    try:
        parameters = {}

        # Get diffusion parameters
        if not process_type or process_type == "diffusion":
            diffusion_query = (
                select(DiffusionSPCSeries.parameter, func.count())
                .where(DiffusionSPCSeries.org_id == org_id)
                .group_by(DiffusionSPCSeries.parameter)
            )
            diffusion_params = db.execute(diffusion_query).all()

            for param, count in diffusion_params:
                if param not in parameters:
                    parameters[param] = {"parameter": param, "processes": [], "total_series": 0}
                parameters[param]["processes"].append("diffusion")
                parameters[param]["total_series"] += count

        # Get CVD parameters
        if not process_type or process_type == "cvd":
            cvd_query = (
                select(CVDSPCSeries.parameter, func.count())
                .where(CVDSPCSeries.org_id == org_id)
                .group_by(CVDSPCSeries.parameter)
            )
            cvd_params = db.execute(cvd_query).all()

            for param, count in cvd_params:
                if param not in parameters:
                    parameters[param] = {"parameter": param, "processes": [], "total_series": 0}
                parameters[param]["processes"].append("cvd")
                parameters[param]["total_series"] += count

        # TODO: Add oxidation, ion, rtp when available

        result = list(parameters.values())
        result.sort(key=lambda x: x["total_series"], reverse=True)

        logger.info(f"Retrieved {len(result)} unique parameters for org {org_id}")
        return result

    except Exception as e:
        logger.exception(f"Error getting available parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))
