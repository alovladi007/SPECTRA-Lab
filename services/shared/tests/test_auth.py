"""
Unit tests for JWT authentication and password hashing.
"""

import pytest
from datetime import datetime, timedelta, timezone
import jwt as pyjwt

from services.shared.auth.jwt import (
    create_access_token,
    create_refresh_token,
    create_token_pair,
    decode_token,
    hash_password,
    verify_password,
    JWT_SECRET,
    JWT_ALGORITHM,
    JWT_ISSUER,
    JWT_AUDIENCE
)
from services.shared.db.models import User, UserRole


class TestJWTTokens:
    """Test JWT token creation and validation."""

    def test_create_access_token(self):
        """Test access token creation with correct claims."""
        payload = {
            "sub": "user-123",
            "org": "org-456",
            "role": "engineer",
            "email": "test@example.com"
        }
        token = create_access_token(payload)

        # Decode token
        decoded = pyjwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER
        )

        # Verify claims
        assert decoded["sub"] == "user-123"
        assert decoded["org"] == "org-456"
        assert decoded["role"] == "engineer"
        assert decoded["email"] == "test@example.com"
        assert decoded["iss"] == JWT_ISSUER
        assert decoded["aud"] == JWT_AUDIENCE
        assert "exp" in decoded
        assert "iat" in decoded

    def test_create_refresh_token(self):
        """Test refresh token creation with longer expiry."""
        payload = {"sub": "user-123", "org": "org-456"}
        token = create_refresh_token(payload)

        decoded = pyjwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER
        )

        assert decoded["sub"] == "user-123"
        assert decoded["org"] == "org-456"

        # Verify refresh token has longer expiry (7 days default)
        iat = datetime.fromtimestamp(decoded["iat"], tz=timezone.utc)
        exp = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        duration = exp - iat
        assert duration >= timedelta(days=6, hours=23)  # Allow some variance

    def test_create_token_pair(self):
        """Test creation of access + refresh token pair."""
        tokens = create_token_pair(
            user_id="user-123",
            org_id="org-456",
            role="pi",
            email="pi@example.com"
        )

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"

        # Verify access token
        access_decoded = decode_token(tokens["access_token"])
        assert access_decoded["sub"] == "user-123"
        assert access_decoded["org"] == "org-456"
        assert access_decoded["role"] == "pi"

    def test_decode_token_valid(self):
        """Test decoding a valid token."""
        payload = {"sub": "user-123", "org": "org-456"}
        token = create_access_token(payload)

        decoded = decode_token(token)
        assert decoded["sub"] == "user-123"
        assert decoded["org"] == "org-456"

    def test_decode_token_expired(self):
        """Test decoding an expired token raises error."""
        payload = {"sub": "user-123"}
        # Create token with negative expiry
        token = create_access_token(payload, expires_delta=timedelta(seconds=-10))

        with pytest.raises(pyjwt.ExpiredSignatureError):
            decode_token(token)

    def test_decode_token_invalid_signature(self):
        """Test decoding token with wrong signature."""
        # Create token with different secret
        token = pyjwt.encode(
            {"sub": "user-123", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            "wrong-secret",
            algorithm=JWT_ALGORITHM
        )

        with pytest.raises(pyjwt.InvalidSignatureError):
            decode_token(token)

    def test_decode_token_missing_required_claims(self):
        """Test decoding token without required claims."""
        # Token without 'sub' claim
        token = pyjwt.encode(
            {
                "exp": datetime.now(timezone.utc) + timedelta(hours=1),
                "iss": JWT_ISSUER,
                "aud": JWT_AUDIENCE
            },
            JWT_SECRET,
            algorithm=JWT_ALGORITHM
        )

        # Should decode successfully - 'sub' is not enforced at decode level
        # but will fail at application level
        decoded = decode_token(token)
        assert "sub" not in decoded


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password(self):
        """Test password hashing produces valid bcrypt hash."""
        password = "MySecurePassword123!"
        hashed = hash_password(password)

        # Bcrypt hashes start with $2b$ or $2a$
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
        assert len(hashed) == 60  # Standard bcrypt hash length

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "MySecurePassword123!"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with wrong password."""
        password = "MySecurePassword123!"
        hashed = hash_password(password)

        assert verify_password("WrongPassword", hashed) is False

    def test_hash_password_different_each_time(self):
        """Test that hashing same password produces different hashes (due to salt)."""
        password = "MySecurePassword123!"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        # Hashes should be different due to random salt
        assert hash1 != hash2

        # But both should verify successfully
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True

    def test_verify_password_empty_password(self):
        """Test verification with empty password."""
        hashed = hash_password("password123")
        assert verify_password("", hashed) is False

    def test_hash_empty_password(self):
        """Test hashing empty password."""
        hashed = hash_password("")
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
        assert verify_password("", hashed) is True


class TestAuthIntegration:
    """Integration tests for auth flow."""

    def test_full_auth_flow_with_user(self, db_session, org1):
        """Test complete auth flow: create user, hash password, generate token."""
        # Create user with hashed password
        password = "engineer123"
        user = User(
            organization_id=org1.id,
            email="newengineer@acme.com",
            full_name="New Engineer",
            role=UserRole.ENGINEER,
            password_hash=hash_password(password),
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        # Verify password
        assert verify_password(password, user.password_hash) is True
        assert verify_password("wrongpass", user.password_hash) is False

        # Generate token pair
        tokens = create_token_pair(
            user_id=str(user.id),
            org_id=str(user.organization_id),
            role=user.role.value,
            email=user.email
        )

        # Decode and verify
        payload = decode_token(tokens["access_token"])
        assert payload["sub"] == str(user.id)
        assert payload["org"] == str(user.organization_id)
        assert payload["role"] == UserRole.ENGINEER.value
        assert payload["email"] == user.email

    def test_token_contains_all_user_context(self, admin_user):
        """Test that token contains all necessary user context."""
        tokens = create_token_pair(
            user_id=str(admin_user.id),
            org_id=str(admin_user.organization_id),
            role=admin_user.role.value,
            email=admin_user.email
        )

        payload = decode_token(tokens["access_token"])

        # Verify all necessary claims for authorization
        assert "sub" in payload  # User ID
        assert "org" in payload  # Organization ID
        assert "role" in payload  # User role
        assert "email" in payload  # Email
        assert "exp" in payload  # Expiration
        assert "iat" in payload  # Issued at
        assert "iss" in payload  # Issuer
        assert "aud" in payload  # Audience

    def test_oidc_user_without_password(self, db_session, org1):
        """Test OIDC user (no password hash) can still get tokens."""
        # Create OIDC user (no password hash)
        user = User(
            organization_id=org1.id,
            email="oidc@acme.com",
            full_name="OIDC User",
            role=UserRole.VIEWER,
            password_hash=None,  # OIDC users don't have passwords
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        # Should still be able to generate tokens
        tokens = create_token_pair(
            user_id=str(user.id),
            org_id=str(user.organization_id),
            role=user.role.value,
            email=user.email
        )

        assert tokens["access_token"]
        assert tokens["refresh_token"]
