"""
Database Connection and Session Management for Diffusion Module
Provides connection pooling, session management, and helper functions for database operations
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import os
from contextlib import contextmanager
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Database connection URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/diffusion"
)

# Create engine with connection pooling
# Use NullPool for development to avoid connection issues
# In production, use default pool or QueuePool
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # For development - no connection pooling
    echo=True if os.getenv("ENVIRONMENT") == "development" else False,
    pool_pre_ping=True,  # Verify connections before using
    connect_args={
        "connect_timeout": 10,
        "options": "-c timezone=utc"
    }
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Session:
    """
    Database session context manager

    Usage:
        with get_db() as db:
            result = db.query(SimulationAudit).all()

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        db.close()


def save_simulation(simulation_data: dict) -> str:
    """
    Save simulation to database

    Args:
        simulation_data: Dictionary with simulation parameters and results

    Returns:
        str: Simulation ID (UUID as string)

    Example:
        sim_id = save_simulation({
            "simulation_type": "diffusion",
            "parameters": {"temperature": 1000, "time": 30},
            "results": {"junction_depth": 250.5},
            "status": "completed"
        })
    """
    from .models import SimulationAudit

    with get_db() as db:
        sim = SimulationAudit(**simulation_data)
        db.add(sim)
        db.flush()
        return str(sim.simulation_id)


def get_simulation(simulation_id: str) -> Optional[Dict]:
    """
    Retrieve simulation from database

    Args:
        simulation_id: UUID string of simulation

    Returns:
        dict: Simulation data or None if not found
    """
    from .models import SimulationAudit

    with get_db() as db:
        sim = db.query(SimulationAudit).filter(
            SimulationAudit.simulation_id == simulation_id
        ).first()

        if not sim:
            return None

        return {
            "simulation_id": str(sim.simulation_id),
            "simulation_type": sim.simulation_type,
            "parameters": sim.parameters,
            "results": sim.results,
            "status": sim.status,
            "created_at": sim.created_at.isoformat() if sim.created_at else None,
            "completed_at": sim.completed_at.isoformat() if sim.completed_at else None,
            "execution_time_ms": sim.execution_time_ms,
        }


def get_recent_simulations(limit: int = 10, simulation_type: Optional[str] = None) -> List[Dict]:
    """
    Get recent simulations

    Args:
        limit: Maximum number of simulations to return
        simulation_type: Filter by type (diffusion, oxidation, calibration)

    Returns:
        List[dict]: List of simulation data
    """
    from .models import SimulationAudit

    with get_db() as db:
        query = db.query(SimulationAudit)

        if simulation_type:
            query = query.filter(SimulationAudit.simulation_type == simulation_type)

        sims = query.order_by(SimulationAudit.created_at.desc()).limit(limit).all()

        return [
            {
                "simulation_id": str(sim.simulation_id),
                "simulation_type": sim.simulation_type,
                "status": sim.status,
                "created_at": sim.created_at.isoformat() if sim.created_at else None,
                "parameters": sim.parameters,
            }
            for sim in sims
        ]


def save_batch_job(job_data: dict) -> str:
    """
    Save batch job to database

    Args:
        job_data: Dictionary with batch job configuration

    Returns:
        str: Job ID (UUID as string)
    """
    from .models import BatchJob

    with get_db() as db:
        job = BatchJob(**job_data)
        db.add(job)
        db.flush()
        return str(job.job_id)


def get_batch_job(job_id: str) -> Optional[Dict]:
    """
    Retrieve batch job from database

    Args:
        job_id: UUID string of batch job

    Returns:
        dict: Batch job data or None if not found
    """
    from .models import BatchJob

    with get_db() as db:
        job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()

        if not job:
            return None

        return {
            "job_id": str(job.job_id),
            "job_name": job.job_name,
            "simulation_type": job.simulation_type,
            "total_simulations": job.total_simulations,
            "completed_simulations": job.completed_simulations,
            "failed_simulations": job.failed_simulations,
            "status": job.status,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "config": job.config,
            "results_summary": job.results_summary,
        }


def update_batch_job_progress(job_id: str, completed: int, failed: int, status: str):
    """
    Update batch job progress

    Args:
        job_id: UUID string of batch job
        completed: Number of completed simulations
        failed: Number of failed simulations
        status: Job status (queued, running, completed, failed)
    """
    from .models import BatchJob

    with get_db() as db:
        job = db.query(BatchJob).filter(BatchJob.job_id == job_id).first()
        if job:
            job.completed_simulations = completed
            job.failed_simulations = failed
            job.status = status


def save_kpi_measurement(kpi_data: dict) -> str:
    """
    Save KPI measurement to database

    Args:
        kpi_data: Dictionary with KPI measurement data

    Returns:
        str: Measurement ID (UUID as string)
    """
    from .models import KPIMeasurement

    with get_db() as db:
        kpi = KPIMeasurement(**kpi_data)
        db.add(kpi)
        db.flush()
        return str(kpi.measurement_id)


def save_spc_violation(violation_data: dict) -> str:
    """
    Save SPC violation to database

    Args:
        violation_data: Dictionary with SPC violation data

    Returns:
        str: Violation ID (UUID as string)
    """
    from .models import SPCViolation

    with get_db() as db:
        violation = SPCViolation(**violation_data)
        db.add(violation)
        db.flush()
        return str(violation.violation_id)


def save_maintenance_recommendation(recommendation_data: dict) -> str:
    """
    Save maintenance recommendation to database

    Args:
        recommendation_data: Dictionary with maintenance recommendation

    Returns:
        str: Recommendation ID (UUID as string)
    """
    from .models import MaintenanceRecommendation

    with get_db() as db:
        rec = MaintenanceRecommendation(**recommendation_data)
        db.add(rec)
        db.flush()
        return str(rec.recommendation_id)


def save_calibration_result(calibration_data: dict) -> str:
    """
    Save calibration result to database

    Args:
        calibration_data: Dictionary with calibration results

    Returns:
        str: Calibration ID (UUID as string)
    """
    from .models import CalibrationResult

    with get_db() as db:
        cal = CalibrationResult(**calibration_data)
        db.add(cal)
        db.flush()
        return str(cal.calibration_id)


def health_check() -> bool:
    """
    Check if database connection is healthy

    Returns:
        bool: True if database is accessible, False otherwise
    """
    try:
        with get_db() as db:
            db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
