"""
deps.py
-------
FastAPI dependencies: the database session and the authenticated principal.

get_session is a transaction boundary. A request either commits every
write it made or none of them -- which is the guarantee the old code could
not offer, since assign_department committed the ticket and the audit log
separately across two connections.

get_principal is the authorization gate that can_access() was supposed to
be. Declaring it on a route is what makes that route require a valid,
unrevoked session for an active user.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from uniflow.core import security
from uniflow.core.errors import AuthenticationError
from uniflow.db.session import get_session_factory
from uniflow.domain import auth
from uniflow.domain.principal import Principal, assert_admin

bearer = HTTPBearer(auto_error=False)


def get_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


def get_principal(
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)] = None,
) -> Principal:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = security.decode_access_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    try:
        # Re-checked against the database on every request, so a logout or
        # a deactivation takes effect immediately rather than whenever the
        # token happens to expire.
        return auth.principal_for(session, int(payload["sub"]), payload["sid"])
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.message
        ) from exc


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def get_admin(principal: PrincipalDep) -> Principal:
    assert_admin(principal)
    return principal


AdminDep = Annotated[Principal, Depends(get_admin)]


def client_agent(user_agent: Annotated[str | None, Header()] = None) -> str | None:
    return user_agent
