"""
services/shared/auth/jwt.py

JWT token handling for SPECTRA-Lab Platform.
Supports both dev mode (HS256 with shared secret) and production mode (RS256 with OIDC/JWKS).
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import jwt, JWTError, jwk
from jose.backends import RSAKey
from passlib.context import CryptContext
import os
import logging
import httpx

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# JWT Settings
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")  # HS256 for dev, RS256 for prod
JWT_ISSUER = os.getenv("JWT_ISSUER", "spectra-lab")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "spectra-lab-api")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# OIDC Settings (for production)
OIDC_ENABLED = os.getenv("OIDC_ENABLED", "false").lower() == "true"
OIDC_JWKS_URL = os.getenv("OIDC_JWKS_URL", "")
OIDC_ISSUER = os.getenv("OIDC_ISSUER", "")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWKS cache
_jwks_cache: Optional[Dict[str, Any]] = None
_jwks_cache_time: Optional[datetime] = None
JWKS_CACHE_TTL = timedelta(hours=1)

# ============================================================================
# Password Utilities
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Bcrypt hash
    
    Returns:
        bool: True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.
    
    Args:
        password: Plain text password
    
    Returns:
        str: Bcrypt hash
    """
    return pwd_context.hash(password)


# ============================================================================
# Token Generation
# ============================================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Payload dict (should include "sub" for user ID)
        expires_delta: Token expiration time
    
    Returns:
        str: Encoded JWT token
    
    Example:
        token = create_access_token(
            data={"sub": str(user.id), "org": str(user.org_id), "role": user.role}
        )
    """
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
    """
    Create JWT refresh token (longer expiry).
    
    Args:
        data: Payload dict
    
    Returns:
        str: Encoded refresh token
    """
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


# ============================================================================
# Token Validation
# ============================================================================

def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT token.
    
    Supports both dev mode (HS256) and production OIDC mode (RS256 with JWKS).
    
    Args:
        token: JWT token string
    
    Returns:
        dict: Decoded payload
    
    Raises:
        JWTError: If token invalid or expired
    """
    if OIDC_ENABLED:
        return decode_oidc_token(token)
    else:
        return decode_dev_token(token)


def decode_dev_token(token: str) -> Dict[str, Any]:
    """
    Decode token in dev mode (HS256 with shared secret).
    
    Args:
        token: JWT token string
    
    Returns:
        dict: Decoded payload
    
    Raises:
        JWTError: If token invalid
    """
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE
        )
        return payload
    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise


def decode_oidc_token(token: str) -> Dict[str, Any]:
    """
    Decode token in OIDC mode (RS256 with JWKS).
    
    Fetches public keys from OIDC provider and validates signature.
    
    Args:
        token: JWT token string
    
    Returns:
        dict: Decoded payload
    
    Raises:
        JWTError: If token invalid
    """
    global _jwks_cache, _jwks_cache_time
    
    # Refresh JWKS cache if expired
    now = datetime.now(timezone.utc)
    if _jwks_cache is None or _jwks_cache_time is None or (now - _jwks_cache_time) > JWKS_CACHE_TTL:
        _jwks_cache = fetch_jwks()
        _jwks_cache_time = now
    
    # Decode header to get key ID
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    
    if not kid:
        raise JWTError("Token missing 'kid' in header")
    
    # Find matching key
    key_data = None
    for key in _jwks_cache.get("keys", []):
        if key.get("kid") == kid:
            key_data = key
            break
    
    if not key_data:
        raise JWTError(f"Key ID '{kid}' not found in JWKS")
    
    # Convert JWK to PEM
    public_key = jwk.construct(key_data)
    
    # Decode and validate
    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer=OIDC_ISSUER,
            audience=JWT_AUDIENCE
        )
        return payload
    except JWTError as e:
        logger.error(f"OIDC token validation failed: {e}")
        raise


def fetch_jwks() -> Dict[str, Any]:
    """
    Fetch JWKS from OIDC provider.
    
    Returns:
        dict: JWKS document
    
    Raises:
        Exception: If fetch fails
    """
    if not OIDC_JWKS_URL:
        raise Exception("OIDC_JWKS_URL not configured")
    
    try:
        response = httpx.get(OIDC_JWKS_URL, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        raise Exception("Could not fetch OIDC public keys")


# ============================================================================
# Token Utilities
# ============================================================================

def create_token_pair(user_id: str, org_id: str, role: str, email: str) -> Dict[str, str]:
    """
    Create access + refresh token pair.
    
    Args:
        user_id: User UUID
        org_id: Organization UUID
        role: User role
        email: User email
    
    Returns:
        dict: {"access_token": ..., "refresh_token": ..., "token_type": "bearer"}
    """
    payload = {
        "sub": user_id,
        "org": org_id,
        "role": role,
        "email": email
    }
    
    access_token = create_access_token(payload)
    refresh_token = create_refresh_token({"sub": user_id, "org": org_id})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


def refresh_access_token(refresh_token: str) -> Dict[str, str]:
    """
    Generate new access token from refresh token.
    
    Args:
        refresh_token: Valid refresh token
    
    Returns:
        dict: New token pair
    
    Raises:
        JWTError: If refresh token invalid
    """
    try:
        payload = decode_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise JWTError("Not a refresh token")
        
        user_id = payload.get("sub")
        org_id = payload.get("org")
        
        # In production, should fetch fresh user data from DB here
        # For now, reuse payload data
        new_access = create_access_token({
            "sub": user_id,
            "org": org_id
        })
        
        return {
            "access_token": new_access,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    except JWTError as e:
        logger.error(f"Refresh token invalid: {e}")
        raise


# ============================================================================
# Development Utilities
# ============================================================================

def generate_dev_token(
    user_id: str,
    org_id: str,
    role: str = "engineer",
    email: str = "dev@example.com"
) -> str:
    """
    Generate dev token for testing (without DB).
    
    Args:
        user_id: User UUID string
        org_id: Organization UUID string
        role: User role
        email: User email
    
    Returns:
        str: JWT token
    """
    return create_access_token({
        "sub": user_id,
        "org": org_id,
        "role": role,
        "email": email
    })


if __name__ == "__main__":
    # Test token generation
    import uuid
    
    user_id = str(uuid.uuid4())
    org_id = str(uuid.uuid4())
    
    tokens = create_token_pair(user_id, org_id, "engineer", "test@demo.lab")
    print("Access Token:", tokens["access_token"])
    print("\nRefresh Token:", tokens["refresh_token"])
    
    # Test decoding
    payload = decode_token(tokens["access_token"])
    print("\nDecoded Payload:", payload)
