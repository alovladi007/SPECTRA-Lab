"""Authentication and RBAC middleware for Process Control APIs.

Implements JWT-based authentication, role-based access control (RBAC), and
organization-level access guards for Ion Implantation and RTP endpoints.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum as PyEnum

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

# JWT configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Auth service URL (for production user validation)
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")


# ============================================================================
# Enums
# ============================================================================

class Role(str, PyEnum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"              # Full access to all resources
    ENGINEER = "engineer"        # Create runs, tune controllers, view all data
    OPERATOR = "operator"        # Start/stop runs, view real-time data
    VIEWER = "viewer"            # Read-only access


class Permission(str, PyEnum):
    """Fine-grained permissions for Ion and RTP operations."""
    # Ion Implantation
    ION_CREATE_RUN = "ion:create_run"
    ION_VIEW_RUN = "ion:view_run"
    ION_CANCEL_RUN = "ion:cancel_run"
    ION_VIEW_TELEMETRY = "ion:view_telemetry"
    ION_SIMULATE = "ion:simulate"
    ION_CALIBRATE = "ion:calibrate"

    # RTP
    RTP_CREATE_RUN = "rtp:create_run"
    RTP_VIEW_RUN = "rtp:view_run"
    RTP_CANCEL_RUN = "rtp:cancel_run"
    RTP_VIEW_TELEMETRY = "rtp:view_telemetry"
    RTP_TUNE_CONTROLLER = "rtp:tune_controller"

    # Jobs
    JOB_VIEW_STATUS = "job:view_status"
    JOB_VIEW_LOGS = "job:view_logs"
    JOB_CANCEL = "job:cancel"
    JOB_RETRY = "job:retry"

    # SPC & VM
    SPC_VIEW_CHARTS = "spc:view_charts"
    SPC_CONFIGURE = "spc:configure"
    VM_VIEW_PREDICTIONS = "vm:view_predictions"
    VM_DEPLOY_MODEL = "vm:deploy_model"


# ============================================================================
# Role-Permission Mapping
# ============================================================================

ROLE_PERMISSIONS: Dict[Role, List[Permission]] = {
    Role.ADMIN: [
        # All permissions
        Permission.ION_CREATE_RUN,
        Permission.ION_VIEW_RUN,
        Permission.ION_CANCEL_RUN,
        Permission.ION_VIEW_TELEMETRY,
        Permission.ION_SIMULATE,
        Permission.ION_CALIBRATE,
        Permission.RTP_CREATE_RUN,
        Permission.RTP_VIEW_RUN,
        Permission.RTP_CANCEL_RUN,
        Permission.RTP_VIEW_TELEMETRY,
        Permission.RTP_TUNE_CONTROLLER,
        Permission.JOB_VIEW_STATUS,
        Permission.JOB_VIEW_LOGS,
        Permission.JOB_CANCEL,
        Permission.JOB_RETRY,
        Permission.SPC_VIEW_CHARTS,
        Permission.SPC_CONFIGURE,
        Permission.VM_VIEW_PREDICTIONS,
        Permission.VM_DEPLOY_MODEL,
    ],

    Role.ENGINEER: [
        # Create and manage runs
        Permission.ION_CREATE_RUN,
        Permission.ION_VIEW_RUN,
        Permission.ION_CANCEL_RUN,
        Permission.ION_VIEW_TELEMETRY,
        Permission.ION_SIMULATE,
        Permission.RTP_CREATE_RUN,
        Permission.RTP_VIEW_RUN,
        Permission.RTP_CANCEL_RUN,
        Permission.RTP_VIEW_TELEMETRY,
        Permission.RTP_TUNE_CONTROLLER,
        Permission.JOB_VIEW_STATUS,
        Permission.JOB_VIEW_LOGS,
        Permission.JOB_CANCEL,
        Permission.SPC_VIEW_CHARTS,
        Permission.SPC_CONFIGURE,
        Permission.VM_VIEW_PREDICTIONS,
    ],

    Role.OPERATOR: [
        # Start/stop runs, view telemetry
        Permission.ION_CREATE_RUN,
        Permission.ION_VIEW_RUN,
        Permission.ION_CANCEL_RUN,
        Permission.ION_VIEW_TELEMETRY,
        Permission.RTP_CREATE_RUN,
        Permission.RTP_VIEW_RUN,
        Permission.RTP_CANCEL_RUN,
        Permission.RTP_VIEW_TELEMETRY,
        Permission.JOB_VIEW_STATUS,
        Permission.SPC_VIEW_CHARTS,
        Permission.VM_VIEW_PREDICTIONS,
    ],

    Role.VIEWER: [
        # Read-only access
        Permission.ION_VIEW_RUN,
        Permission.ION_VIEW_TELEMETRY,
        Permission.RTP_VIEW_RUN,
        Permission.RTP_VIEW_TELEMETRY,
        Permission.JOB_VIEW_STATUS,
        Permission.SPC_VIEW_CHARTS,
        Permission.VM_VIEW_PREDICTIONS,
    ],
}


# ============================================================================
# Data Models
# ============================================================================

class User(BaseModel):
    """Authenticated user information."""
    user_id: str
    email: str
    org_id: str
    role: Role
    permissions: List[Permission] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TokenData(BaseModel):
    """JWT token payload."""
    user_id: str
    email: str
    org_id: str
    role: str
    exp: Optional[int] = None


# ============================================================================
# JWT Token Utilities
# ============================================================================

def create_access_token(
    user_id: str,
    email: str,
    org_id: str,
    role: Role,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token.

    Args:
        user_id: User ID
        email: User email
        org_id: Organization ID
        role: User role
        expires_delta: Token expiration time (default: 24 hours)

    Returns:
        JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(hours=JWT_EXPIRATION_HOURS)

    expire = datetime.utcnow() + expires_delta

    # Handle both Role enum and string values
    role_value = role.value if isinstance(role, Role) else role

    to_encode = {
        "user_id": user_id,
        "email": email,
        "org_id": org_id,
        "role": role_value,
        "exp": expire,
    }

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """Decode and validate JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData with user information

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        user_id: str = payload.get("user_id")
        email: str = payload.get("email")
        org_id: str = payload.get("org_id")
        role: str = payload.get("role")

        if user_id is None or email is None or org_id is None or role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return TokenData(
            user_id=user_id,
            email=email,
            org_id=org_id,
            role=role,
            exp=payload.get("exp")
        )

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================================
# FastAPI Security Dependencies
# ============================================================================

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user from JWT token.

    This is the main authentication dependency for protected endpoints.

    Usage:
        @app.get("/api/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user_id": user.user_id}

    Args:
        credentials: HTTP Bearer token from Authorization header

    Returns:
        User object with permissions

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    token_data = decode_token(token)

    # Map role to permissions
    role = Role(token_data.role)
    permissions = ROLE_PERMISSIONS.get(role, [])

    return User(
        user_id=token_data.user_id,
        email=token_data.email,
        org_id=token_data.org_id,
        role=role,
        permissions=permissions,
    )


async def get_optional_user(
    authorization: Optional[str] = Header(None)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise.

    For endpoints that optionally use authentication.

    Args:
        authorization: Optional Authorization header

    Returns:
        User object if authenticated, None otherwise
    """
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    token = authorization[7:]  # Remove "Bearer " prefix

    try:
        token_data = decode_token(token)
        role = Role(token_data.role)
        permissions = ROLE_PERMISSIONS.get(role, [])

        return User(
            user_id=token_data.user_id,
            email=token_data.email,
            org_id=token_data.org_id,
            role=role,
            permissions=permissions,
        )
    except HTTPException:
        return None


# ============================================================================
# Permission Checking
# ============================================================================

def require_permission(permission: Permission):
    """Dependency factory for permission-based access control.

    Usage:
        @app.post("/api/ion/runs")
        async def create_ion_run(
            user: User = Depends(require_permission(Permission.ION_CREATE_RUN))
        ):
            return {"status": "created"}

    Args:
        permission: Required permission

    Returns:
        FastAPI dependency function
    """
    async def permission_checker(user: User = Depends(get_current_user)) -> User:
        if permission not in user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required permission: {permission.value}",
            )
        return user

    return permission_checker


def require_any_permission(*permissions: Permission):
    """Dependency factory requiring ANY of the specified permissions.

    Usage:
        @app.get("/api/runs/{run_id}")
        async def get_run(
            user: User = Depends(require_any_permission(
                Permission.ION_VIEW_RUN,
                Permission.RTP_VIEW_RUN
            ))
        ):
            return {"run": "data"}

    Args:
        *permissions: List of acceptable permissions

    Returns:
        FastAPI dependency function
    """
    async def permission_checker(user: User = Depends(get_current_user)) -> User:
        if not any(p in user.permissions for p in permissions):
            perm_names = [p.value for p in permissions]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required one of: {perm_names}",
            )
        return user

    return permission_checker


def require_all_permissions(*permissions: Permission):
    """Dependency factory requiring ALL of the specified permissions.

    Args:
        *permissions: List of required permissions

    Returns:
        FastAPI dependency function
    """
    async def permission_checker(user: User = Depends(get_current_user)) -> User:
        missing = [p for p in permissions if p not in user.permissions]
        if missing:
            perm_names = [p.value for p in missing]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Missing permissions: {perm_names}",
            )
        return user

    return permission_checker


def require_role(min_role: Role):
    """Dependency factory for role-based access control.

    Checks role hierarchy: ADMIN > ENGINEER > OPERATOR > VIEWER

    Usage:
        @app.post("/api/spc/configure")
        async def configure_spc(
            user: User = Depends(require_role(Role.ENGINEER))
        ):
            return {"status": "configured"}

    Args:
        min_role: Minimum required role

    Returns:
        FastAPI dependency function
    """
    role_hierarchy = {
        Role.ADMIN: 4,
        Role.ENGINEER: 3,
        Role.OPERATOR: 2,
        Role.VIEWER: 1,
    }

    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if role_hierarchy.get(user.role, 0) < role_hierarchy.get(min_role, 0):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {min_role.value} or higher",
            )
        return user

    return role_checker


# ============================================================================
# Organization Access Control
# ============================================================================

def require_org_access(resource_org_id: str, user: User) -> None:
    """Check if user has access to a resource in a specific organization.

    Raises HTTPException if access denied.

    Usage:
        @app.get("/api/ion/runs/{run_id}")
        async def get_run(
            run_id: str,
            user: User = Depends(get_current_user)
        ):
            run = get_run_from_db(run_id)
            require_org_access(run.org_id, user)
            return run

    Args:
        resource_org_id: Organization ID of the resource
        user: Authenticated user

    Raises:
        HTTPException: If user's org doesn't match resource org
    """
    if user.org_id != resource_org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Resource belongs to a different organization.",
        )


def require_org_access_multi(resource_org_ids: List[str], user: User) -> None:
    """Check if user has access to resources in specific organizations.

    User must belong to one of the organizations.

    Args:
        resource_org_ids: List of organization IDs
        user: Authenticated user

    Raises:
        HTTPException: If user's org doesn't match any resource org
    """
    if user.org_id not in resource_org_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Resources belong to different organizations.",
        )


# ============================================================================
# Development Utilities
# ============================================================================

class DevAuth:
    """Development-only authentication bypass.

    WARNING: Only use in development mode. Never in production.
    """

    @staticmethod
    def create_dev_user(
        user_id: str = "dev-user-1",
        email: str = "dev@example.com",
        org_id: str = "org-1",
        role: Role = Role.ADMIN
    ) -> User:
        """Create a dev user with specified role.

        Args:
            user_id: User ID
            email: User email
            org_id: Organization ID
            role: User role (default: ADMIN)

        Returns:
            User object with full permissions
        """
        return User(
            user_id=user_id,
            email=email,
            org_id=org_id,
            role=role,
            permissions=ROLE_PERMISSIONS.get(role, []),
        )

    @staticmethod
    def create_dev_token(
        user_id: str = "dev-user-1",
        email: str = "dev@example.com",
        org_id: str = "org-1",
        role: Role = Role.ADMIN
    ) -> str:
        """Create a development JWT token.

        Args:
            user_id: User ID
            email: User email
            org_id: Organization ID
            role: User role (default: ADMIN)

        Returns:
            JWT token string
        """
        return create_access_token(user_id, email, org_id, role)


# ============================================================================
# API Key Authentication (for service-to-service)
# ============================================================================

class APIKeyAuth:
    """API key authentication for service-to-service communication."""

    # In production, store in database
    _api_keys: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_api_key(
        cls,
        api_key: str,
        service_name: str,
        org_id: str,
        permissions: List[Permission]
    ) -> None:
        """Register an API key for a service.

        Args:
            api_key: API key string
            service_name: Name of the service
            org_id: Organization ID
            permissions: List of permissions
        """
        cls._api_keys[api_key] = {
            "service_name": service_name,
            "org_id": org_id,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
        }

    @classmethod
    def validate_api_key(cls, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return service info.

        Args:
            api_key: API key string

        Returns:
            Service info dict or None if invalid
        """
        return cls._api_keys.get(api_key)


async def get_service_auth(
    x_api_key: Optional[str] = Header(None)
) -> Optional[Dict[str, Any]]:
    """Dependency for API key authentication.

    Usage:
        @app.post("/api/internal/callback")
        async def internal_callback(
            service: Dict = Depends(get_service_auth)
        ):
            if not service:
                raise HTTPException(401, "Invalid API key")
            return {"status": "ok"}

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        Service info dict or None
    """
    if not x_api_key:
        return None

    return APIKeyAuth.validate_api_key(x_api_key)


# ============================================================================
# Audit Logging
# ============================================================================

class AuditLogger:
    """Audit logging for sensitive operations.

    In production, send to centralized logging system.
    """

    @staticmethod
    def log_access(
        user: User,
        resource_type: str,
        resource_id: str,
        action: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log access attempt.

        Args:
            user: User making the request
            resource_type: Type of resource (e.g., "ion_run", "rtp_run")
            resource_id: Resource identifier
            action: Action attempted (e.g., "create", "view", "cancel")
            success: Whether the action succeeded
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user.user_id,
            "org_id": user.org_id,
            "role": user.role.value,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "success": success,
            "metadata": metadata or {},
        }

        # In production: send to Elasticsearch, CloudWatch, etc.
        print(f"[AUDIT] {log_entry}")


# Export
__all__ = [
    "Role",
    "Permission",
    "User",
    "TokenData",
    "ROLE_PERMISSIONS",
    "create_access_token",
    "decode_token",
    "get_current_user",
    "get_optional_user",
    "require_permission",
    "require_any_permission",
    "require_all_permissions",
    "require_role",
    "require_org_access",
    "require_org_access_multi",
    "DevAuth",
    "APIKeyAuth",
    "get_service_auth",
    "AuditLogger",
]
