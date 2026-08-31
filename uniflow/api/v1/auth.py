"""Authentication endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from uniflow.api.deps import PrincipalDep, SessionDep, client_agent
from uniflow.api.schemas import LoginRequest, RefreshRequest, TokenOut, UserOut
from uniflow.db.models import User
from uniflow.domain import auth as auth_service

router = APIRouter(tags=["auth"])

AgentDep = Annotated[str | None, Depends(client_agent)]


@router.post("/auth/login", response_model=TokenOut)
def login(payload: LoginRequest, session: SessionDep, user_agent: AgentDep):
    pair = auth_service.login(
        session, payload.username, payload.password, user_agent=user_agent
    )
    return TokenOut(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.expires_in,
        user=UserOut.model_validate(pair.user),
    )


@router.post("/auth/refresh", response_model=TokenOut)
def refresh(payload: RefreshRequest, session: SessionDep, user_agent: AgentDep):
    pair = auth_service.refresh(session, payload.refresh_token, user_agent=user_agent)
    return TokenOut(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.expires_in,
        user=UserOut.model_validate(pair.user),
    )


@router.post("/auth/logout", status_code=204)
def logout(session: SessionDep, principal: PrincipalDep, everywhere: bool = False):
    """Revokes this session, or every session for the user.

    There was no equivalent before: signing out meant closing the browser
    tab, and nothing on the server changed.
    """
    if everywhere or principal.session_id is None:
        auth_service.logout_everywhere(session, principal.user_id)
    else:
        # The session id travels in the access token, so signing out does
        # not require handing the refresh token back.
        auth_service.logout(session, principal.session_id)


@router.get("/me", response_model=UserOut)
def me(session: SessionDep, principal: PrincipalDep):
    return UserOut.model_validate(session.get(User, principal.user_id))
