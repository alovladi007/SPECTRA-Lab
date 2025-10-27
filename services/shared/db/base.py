"""
services/shared/db/base.py

Database connection and session management for SPECTRA-Lab Platform.
Provides SQLAlchemy engine, session factory, and base declarative class.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool
from contextlib import contextmanager
import os
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://spectra:spectra@localhost:5432/spectra"
)

# Engine configuration
ENGINE_CONFIG = {
    "echo": os.getenv("SQL_ECHO", "false").lower() == "true",
    "pool_pre_ping": True,  # Verify connections before using
    "pool_recycle": 3600,   # Recycle connections after 1 hour
}

# Use QueuePool for production, NullPool for testing
if os.getenv("TESTING", "false").lower() == "true":
    ENGINE_CONFIG["poolclass"] = NullPool
else:
    ENGINE_CONFIG["pool_size"] = int(os.getenv("DB_POOL_SIZE", "20"))
    ENGINE_CONFIG["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "40"))
    ENGINE_CONFIG["poolclass"] = QueuePool

# ============================================================================
# Engine & Session Factory
# ============================================================================

engine = create_engine(DATABASE_URL, **ENGINE_CONFIG)

# Configure statement timeout (30 seconds default)
@event.listens_for(engine, "connect")
def set_postgres_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("SET statement_timeout = 30000")  # 30 seconds
    cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Prevent lazy-load errors after commit
)

# ============================================================================
# Declarative Base
# ============================================================================

Base = declarative_base()

# ============================================================================
# Session Management
# ============================================================================

def get_db() -> Session:
    """
    FastAPI dependency for database sessions.
    
    Usage:
        @app.get("/items")
        async def list_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    
    Yields:
        Session: SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def session_scope():
    """
    Context manager for standalone session usage.
    
    Usage:
        with session_scope() as session:
            user = session.query(User).first()
            user.name = "Updated"
            session.commit()
    
    Yields:
        Session: SQLAlchemy session with automatic commit/rollback
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Transaction error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


# ============================================================================
# Health Check
# ============================================================================

def check_database_health() -> dict:
    """
    Verify database connectivity and basic operations.
    
    Returns:
        dict: Health status with connection info
    """
    try:
        with session_scope() as session:
            result = session.execute("SELECT 1").scalar()
            if result != 1:
                raise Exception("Unexpected query result")
        
        return {
            "status": "healthy",
            "database": "postgresql",
            "connection": "active"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "postgresql",
            "error": str(e)
        }


# ============================================================================
# Testing Helpers
# ============================================================================

def create_all_tables():
    """Create all tables (for testing only)."""
    Base.metadata.create_all(bind=engine)


def drop_all_tables():
    """Drop all tables (for testing only)."""
    Base.metadata.drop_all(bind=engine)


# ============================================================================
# Cleanup
# ============================================================================

def dispose_engine():
    """Dispose of the engine connection pool."""
    engine.dispose()
    logger.info("Database engine disposed")
