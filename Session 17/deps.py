"""
services/shared/db/deps.py

FastAPI dependencies for authentication, authorization, and database sessions.
Provides:
- Database session injection
- Current user resolution from JWT
- Role-based access control guards
- Organization scoping filters
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import logging

from .base import SessionLocal
from .models import User, Organization, UserRole

logger = logging.getLogger(__name__)

# ============================================================================
# Security Schemes
# ============================================================================

security = HTTPBearer(auto_error=False)

# ============================================================================
# Database Session Dependency
# ============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    Provide database session.
    
    Usage:
        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    
    Yields:
        Session: SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# ============================================================================
# Authentication Dependencies
# ============================================================================

def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Extract current user from JWT token (optional).
    
    Returns None if no token provided or invalid.
    Use for public endpoints that have optional auth.
    
    Args:
        credentials: Bearer token credentials
        db: Database session
    
    Returns:
        User or None
    """
    if not credentials:
        return None
    
    try:
        from services.shared.auth.jwt import decode_token
        payload = decode_token(credentials.credentials)
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None
    
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Extract current user from JWT token (required).
    
    Raises 401 if token missing or invalid.
    Use for protected endpoints.
    
    Args:
        credentials: Bearer token credentials
        db: Database session
    
    Returns:
        User: Authenticated user
    
    Raises:
        HTTPException: 401 if authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        from services.shared.auth.jwt import decode_token
        payload = decode_token(credentials.credentials)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return user


# ============================================================================
# Role-Based Access Control
# ============================================================================

class RoleGuard:
    """
    Dependency class for role-based access control.
    
    Usage:
        @router.post("/recipes/approve")
        def approve_recipe(
            current_user: User = Depends(RoleGuard(["admin", "pi"]))
        ):
            # Only admins and PIs can access this endpoint
            pass
    """
    
    def __init__(self, allowed_roles: list[str]):
        """
        Initialize role guard.
        
        Args:
            allowed_roles: List of role strings (e.g., ["admin", "pi"])
        """
        self.allowed_roles = [r.lower() for r in allowed_roles]
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        """
        Check if user has required role.
        
        Args:
            current_user: Authenticated user
        
        Returns:
            User: User if authorized
        
        Raises:
            HTTPException: 403 if user lacks required role
        """
        if current_user.role.value.lower() not in self.allowed_roles:
            logger.warning(
                f"User {current_user.email} (role={current_user.role.value}) "
                f"attempted to access endpoint requiring roles: {self.allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(self.allowed_roles)}"
            )
        return current_user


# Convenience functions for common roles

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    return RoleGuard(["admin"])(current_user)


def require_pi_or_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require PI or admin role."""
    return RoleGuard(["admin", "pi"])(current_user)


def require_engineer_or_above(current_user: User = Depends(get_current_user)) -> User:
    """Require engineer, PI, or admin role."""
    return RoleGuard(["admin", "pi", "engineer"])(current_user)


def require_technician_or_above(current_user: User = Depends(get_current_user)) -> User:
    """Require technician, engineer, PI, or admin role."""
    return RoleGuard(["admin", "pi", "engineer", "technician"])(current_user)


# ============================================================================
# Organization Scoping
# ============================================================================

class OrgSession:
    """
    Wrapper for database session with automatic org filtering.
    
    Usage:
        @router.get("/samples")
        def list_samples(org_session: OrgSession = Depends(get_org_session)):
            # Automatically filters by current user's org
            return org_session.query(Sample).all()
    """
    
    def __init__(self, session: Session, org_id: str):
        """
        Initialize org session.
        
        Args:
            session: SQLAlchemy session
            org_id: Organization UUID
        """
        self._session = session
        self.org_id = org_id
    
    def query(self, *args, **kwargs):
        """
        Query with automatic org_id filtering.
        
        Returns:
            Query: SQLAlchemy query object
        """
        query = self._session.query(*args, **kwargs)
        # Automatically filter by org_id if model has the attribute
        if args and hasattr(args[0], 'organization_id'):
            query = query.filter(args[0].organization_id == self.org_id)
        return query
    
    def add(self, instance):
        """Add instance to session."""
        return self._session.add(instance)
    
    def commit(self):
        """Commit transaction."""
        return self._session.commit()
    
    def rollback(self):
        """Rollback transaction."""
        return self._session.rollback()
    
    def flush(self):
        """Flush changes."""
        return self._session.flush()
    
    def refresh(self, instance):
        """Refresh instance."""
        return self._session.refresh(instance)
    
    def delete(self, instance):
        """Delete instance."""
        return self._session.delete(instance)


def get_org_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> OrgSession:
    """
    Get org-scoped database session.
    
    Automatically filters queries by current user's organization.
    
    Args:
        current_user: Authenticated user
        db: Database session
    
    Returns:
        OrgSession: Organization-scoped session wrapper
    """
    return OrgSession(db, str(current_user.organization_id))


# ============================================================================
# Calibration Validation
# ============================================================================

def check_instrument_calibration(
    instrument_id: str,
    db: Session
) -> dict:
    """
    Check if instrument has valid calibration.
    
    Args:
        instrument_id: Instrument UUID
        db: Database session
    
    Returns:
        dict: Calibration status with details
        
    Example:
        {
            "valid": False,
            "status": "expired",
            "latest_calibration": {
                "expires_at": "2024-10-01T00:00:00Z",
                "certificate_id": "CAL-12345"
            }
        }
    """
    from .models import Calibration, CalibrationStatus
    from datetime import datetime, timezone
    
    # Get latest calibration
    latest = (
        db.query(Calibration)
        .filter(Calibration.instrument_id == instrument_id)
        .order_by(Calibration.issued_at.desc())
        .first()
    )
    
    if not latest:
        return {
            "valid": False,
            "status": "no_calibration",
            "latest_calibration": None
        }
    
    now = datetime.now(timezone.utc)
    valid = latest.expires_at > now and latest.status == CalibrationStatus.VALID
    
    return {
        "valid": valid,
        "status": "valid" if valid else "expired",
        "latest_calibration": {
            "id": str(latest.id),
            "issued_at": latest.issued_at.isoformat(),
            "expires_at": latest.expires_at.isoformat(),
            "certificate_id": latest.certificate_id
        }
    }


def require_valid_calibration(instrument_id: str, db: Session = Depends(get_db)):
    """
    Dependency to enforce valid calibration.
    
    Usage:
        @router.post("/runs")
        def create_run(
            data: RunCreate,
            calibration_ok: None = Depends(
                lambda db=Depends(get_db): require_valid_calibration(data.instrument_id, db)
            )
        ):
            # Run creation only proceeds if calibration valid
            pass
    
    Args:
        instrument_id: Instrument UUID
        db: Database session
    
    Raises:
        HTTPException: 409 if calibration expired or missing
    """
    status = check_instrument_calibration(instrument_id, db)
    
    if not status["valid"]:
        if status["status"] == "no_calibration":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "no_calibration",
                    "message": "No calibration certificate on record for this instrument",
                    "instrument_id": instrument_id
                }
            )
        else:
            cal = status["latest_calibration"]
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "calibration_expired",
                    "message": "Instrument calibration has expired",
                    "instrument_id": instrument_id,
                    "expires_at": cal["expires_at"],
                    "certificate_id": cal["certificate_id"]
                }
            )


# ============================================================================
# Pagination Helpers
# ============================================================================

class PaginationParams:
    """
    Pagination parameters.
    
    Usage:
        @router.get("/items")
        def list_items(
            pagination: PaginationParams = Depends(),
            db: Session = Depends(get_db)
        ):
            query = db.query(Item)
            items = pagination.paginate(query)
            return items
    """
    
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        max_limit: int = 1000
    ):
        """
        Initialize pagination params.
        
        Args:
            skip: Number of records to skip
            limit: Number of records to return
            max_limit: Maximum allowed limit
        """
        self.skip = max(0, skip)
        self.limit = min(max(1, limit), max_limit)
    
    def paginate(self, query):
        """
        Apply pagination to query.
        
        Args:
            query: SQLAlchemy query
        
        Returns:
            list: Paginated results
        """
        return query.offset(self.skip).limit(self.limit).all()
    
    def metadata(self, total: int) -> dict:
        """
        Generate pagination metadata.
        
        Args:
            total: Total number of records
        
        Returns:
            dict: Pagination metadata
        """
        return {
            "skip": self.skip,
            "limit": self.limit,
            "total": total,
            "page": (self.skip // self.limit) + 1 if self.limit > 0 else 1,
            "pages": (total + self.limit - 1) // self.limit if self.limit > 0 else 1
        }
