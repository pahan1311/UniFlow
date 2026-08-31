"""
security.py
-----------
Password hashing and token issuance.

Two deliberate compatibility choices:

1. Existing PBKDF2 hashes ('salt_hex$hash_hex', 120k iterations) written by
   the old app/auth.py still verify. On a successful login against a legacy
   hash the password is transparently rehashed with Argon2id, so the estate
   migrates itself without a forced reset.
2. Argon2id is used for everything new. It is the current password-hashing
   recommendation and is memory-hard, which PBKDF2 is not.

Sessions replace `st.session_state["user"]`, which had no expiry, no
rotation and no way to revoke a login.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from uniflow.core.settings import get_settings

_LEGACY_ITERATIONS = 120_000
_ph = PasswordHasher()

ALGORITHM = "HS256"


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Argon2id hash for storage."""
    return _ph.hash(password)


def legacy_hash_password(password: str, salt: bytes | None = None) -> str:
    """The old PBKDF2 format. Kept so existing tests and fixtures still pass."""
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _LEGACY_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def _verify_legacy(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, AttributeError):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _LEGACY_ITERATIONS)
    return hmac.compare_digest(candidate.hex(), hash_hex)


def verify_password(password: str, stored: str) -> tuple[bool, bool]:
    """Verifies a password against either hash format.

    Returns (is_valid, needs_rehash). needs_rehash is True for a legacy
    PBKDF2 hash or an Argon2 hash whose parameters are now out of date;
    the caller stores a fresh hash on the next successful login.
    """
    if not stored:
        return False, False

    if stored.startswith("$argon2"):
        try:
            _ph.verify(stored, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False, False
        return True, _ph.check_needs_rehash(stored)

    if _verify_legacy(password, stored):
        return True, True
    return False, False


def dummy_verify() -> None:
    """Burns roughly one hash's worth of time.

    Called when a username does not exist so that a missing account and a
    wrong password take comparable time, which blocks trivial user
    enumeration. The old authenticate() did the same thing with PBKDF2.
    """
    try:
        _ph.verify(
            "$argon2id$v=19$m=65536,t=3,p=4$"
            "c29tZXNhbHRzb21lc2FsdA$8pOKLQCXOBhFVBnJhKX3gYVvVMPqTQVXKvBLSJq6TCE",
            "not-the-password",
        )
    except Exception:  # noqa: BLE001 - the point is the work, not the result
        pass


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def new_refresh_token() -> tuple[uuid.UUID, str]:
    """Returns (session_id, opaque secret) for a new refresh token.

    Only the session id is stored; the secret is hashed. A stolen database
    therefore does not yield usable refresh tokens.
    """
    return uuid.uuid4(), secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_access_token(*, user_id: int, username: str, role: str,
                        department_id: int | None, session_id: uuid.UUID) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "dept": department_id,
        "sid": str(session_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_minutes),
        "typ": "access",
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises jwt.PyJWTError on anything invalid or expired."""
    settings = get_settings()
    payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    if payload.get("typ") != "access":
        raise jwt.InvalidTokenError("not an access token")
    return payload
