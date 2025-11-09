"""
Authentication endpoints for LIMS service
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from jose import jwt
import os

# JWT Configuration (copied from services/shared/auth/jwt.py to avoid import issues)
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ISSUER = os.getenv("JWT_ISSUER", "spectra-lab")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "spectra-lab-api")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token (longer expiry)."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_token_pair(user_id: str, org_id: str, role: str, email: str) -> Dict[str, str]:
    """Create access + refresh token pair."""
    payload = {
        "sub": user_id,
        "org_id": org_id,
        "role": role,
        "email": email
    }

    access_token = create_access_token(payload)
    refresh_token = create_refresh_token({"sub": user_id, "org_id": org_id})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# Demo users (in production, these would come from database)
DEMO_USERS = {
    "admin@acme.com": {
        "id": "00000000-0000-0000-0000-000000000001",
        "email": "admin@acme.com",
        "password": "admin123",
        "role": "admin",
        "org_id": "10000000-0000-0000-0000-000000000001",
    },
    "pi@acme.com": {
        "id": "00000000-0000-0000-0000-000000000002",
        "email": "pi@acme.com",
        "password": "pi123",
        "role": "pi",
        "org_id": "10000000-0000-0000-0000-000000000001",
    },
    "engineer@acme.com": {
        "id": "00000000-0000-0000-0000-000000000003",
        "email": "engineer@acme.com",
        "password": "eng123",
        "role": "engineer",
        "org_id": "10000000-0000-0000-0000-000000000001",
    },
    "tech@acme.com": {
        "id": "00000000-0000-0000-0000-000000000004",
        "email": "tech@acme.com",
        "password": "tech123",
        "role": "technician",
        "org_id": "10000000-0000-0000-0000-000000000001",
    },
    "viewer@acme.com": {
        "id": "00000000-0000-0000-0000-000000000005",
        "email": "viewer@acme.com",
        "password": "view123",
        "role": "viewer",
        "org_id": "10000000-0000-0000-0000-000000000001",
    },
}


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint - returns JWT tokens

    Demo credentials:
    - admin@acme.com / admin123
    - pi@acme.com / pi123
    - engineer@acme.com / eng123
    - tech@acme.com / tech123
    - viewer@acme.com / view123
    """
    # Check if user exists
    user = DEMO_USERS.get(form_data.username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify password
    if user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create JWT tokens
    tokens = create_token_pair(
        user_id=user["id"],
        org_id=user["org_id"],
        role=user["role"],
        email=user["email"],
    )

    return tokens


@router.get("/me")
async def get_current_user():
    """Get current user info (requires valid JWT token)"""
    # In production, this would validate the JWT and return user info
    # For now, return a placeholder
    return {"message": "User info endpoint - requires JWT validation"}
