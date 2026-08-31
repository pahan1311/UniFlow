"""
auth.py
-------
Login, token refresh and logout.

The project claims to route inquiries to *department* dashboards, so
department isolation has to be enforced rather than merely displayed.
That enforcement lives in domain/principal.py; this module is about
establishing who the caller is in the first place.

What the old design had: a password check, then `st.session_state["user"]
= record`. No expiry, no rotation, no revocation, no lockout, no record
that a login had ever happened, and no rate limiting.

Added here: account lockout after repeated failures, rotating refresh
tokens (a refresh consumes the old token and issues a new one, so a
stolen token is usable at most once before the theft becomes visible),
and transparent rehashing of legacy PBKDF2 passwords.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa

from uniflow.core import security
from uniflow.core.errors import AuthenticationError
from uniflow.core.settings import get_settings
from uniflow.db.models import Session as SessionRow
from uniflow.db.models import User
from uniflow.domain.principal import Principal


def _parse_session_id(value: str) -> uuid.UUID:
    """Session ids cross two boundaries as plain strings: inside the JWT
    payload and inside the 'session_id.secret' refresh token. sa.Uuid
    expects a real UUID object on the way back in, so every read-side
    lookup goes through here rather than handing session.get() a string.
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise AuthenticationError("Session identifier is not valid") from None


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int
    user: User


def _issue(session, user: User, user_agent: str | None) -> TokenPair:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    session_id, secret = security.new_refresh_token()

    session.add(SessionRow(
        id=session_id,
        user_id=user.id,
        token_hash=security.hash_token(secret),
        issued_at=now,
        expires_at=now + timedelta(days=settings.refresh_token_days),
        user_agent=(user_agent or "")[:400] or None,
    ))
    access = security.create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role.value,
        department_id=user.department_id,
        session_id=session_id,
    )
    session.flush()
    return TokenPair(
        access_token=access,
        # The session id travels with the secret so refresh is a primary
        # key lookup rather than a scan over every live session's hash.
        refresh_token=f"{session_id}.{secret}",
        expires_in=settings.access_token_minutes * 60,
        user=user,
    )


def login(session, username: str, password: str, *, user_agent: str | None = None) -> TokenPair:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    username = (username or "").strip().lower()

    user = session.execute(
        sa.select(User).where(User.username == username)
    ).scalar_one_or_none()

    if user is None:
        # Hash anyway so a missing username and a wrong password take
        # roughly the same time (avoids trivial user enumeration).
        security.dummy_verify()
        raise AuthenticationError("Incorrect username or password")

    if user.locked_until and user.locked_until > now:
        remaining = int((user.locked_until - now).total_seconds() // 60) + 1
        raise AuthenticationError(
            f"Account locked after repeated failed sign-ins. Try again in {remaining} minutes."
        )

    if not user.is_active:
        raise AuthenticationError("This account has been deactivated")

    valid, needs_rehash = security.verify_password(password or "", user.password_hash)
    if not valid:
        user.failed_login_count += 1
        if user.failed_login_count >= settings.max_failed_logins:
            user.locked_until = now + timedelta(minutes=settings.lockout_minutes)
            user.failed_login_count = 0
        session.flush()
        raise AuthenticationError("Incorrect username or password")

    if needs_rehash:
        # A successful login is the only moment the plaintext exists, so
        # it is the only moment a legacy PBKDF2 hash can be upgraded.
        user.password_hash = security.hash_password(password)

    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = now
    session.flush()
    return _issue(session, user, user_agent)


def refresh(session, refresh_token: str, *, user_agent: str | None = None) -> TokenPair:
    """Consumes a refresh token and issues a new pair.

    Rotation, not reuse: the presented token is revoked as part of the
    same transaction that issues its replacement.
    """
    now = datetime.now(timezone.utc)
    try:
        session_id, secret = (refresh_token or "").split(".", 1)
    except ValueError:
        raise AuthenticationError("Malformed refresh token") from None

    row = session.get(SessionRow, _parse_session_id(session_id))
    if row is None or row.revoked_at is not None or row.expires_at <= now:
        raise AuthenticationError("Refresh token is not valid")

    import hmac

    if not hmac.compare_digest(row.token_hash, security.hash_token(secret)):
        # Presenting a valid session id with the wrong secret is a strong
        # signal of a stolen or guessed token, so the session dies.
        row.revoked_at = now
        session.flush()
        raise AuthenticationError("Refresh token is not valid")

    user = session.get(User, row.user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("This account has been deactivated")

    row.revoked_at = now
    session.flush()
    return _issue(session, user, user_agent)


def logout(session, session_id: str) -> None:
    row = session.get(SessionRow, _parse_session_id(session_id))
    if row is not None and row.revoked_at is None:
        row.revoked_at = datetime.now(timezone.utc)
        session.flush()


def logout_everywhere(session, user_id: int) -> int:
    result = session.execute(
        sa.update(SessionRow)
        .where(SessionRow.user_id == user_id, SessionRow.revoked_at.is_(None))
        .values(revoked_at=datetime.now(timezone.utc))
    )
    session.flush()
    return result.rowcount


def principal_for(session, user_id: int, session_id: str) -> Principal:
    """Resolves a decoded access token to a live principal.

    The session row is re-checked on every request, which is what makes
    logout and deactivation take effect within the access token's lifetime
    instead of whenever the user happens to close the browser.
    """
    row = session.get(SessionRow, _parse_session_id(session_id))
    now = datetime.now(timezone.utc)
    if row is None or row.revoked_at is not None or row.expires_at <= now:
        raise AuthenticationError("Session has ended. Please sign in again.")

    user = session.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("This account has been deactivated")
    return Principal.from_user(user, session_id=str(row.id))
