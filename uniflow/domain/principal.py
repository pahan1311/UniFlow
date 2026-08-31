"""
principal.py
------------
Who is acting, and what they are allowed to touch.

Authorization used to be `scoped(df)`: a pandas filter applied *after*
fetch_all() had already read every row of every department into memory.
can_access() existed but was called from nothing except a test, so every
write handler took a raw inquiry id and mutated it with no ownership check
at all.

A Principal carries the department scope as data, and
`visible_department_ids` is applied as a SQL predicate before rows leave
the database. `assert_can_act_on` is the write-side counterpart, and every
mutating service method calls it.
"""

from __future__ import annotations

from dataclasses import dataclass

from uniflow.core.enums import Role
from uniflow.core.errors import PermissionDeniedError
from uniflow.db.models import Department, Inquiry, User


@dataclass(frozen=True)
class Principal:
    """The authenticated caller, resolved once per request."""

    user_id: int
    username: str
    role: Role
    department_id: int | None
    display_name: str = ""
    # Which login this request belongs to, so a caller can revoke exactly
    # the session it is using without handing back the refresh token.
    session_id: str | None = None

    @classmethod
    def from_user(cls, user: User, session_id: str | None = None) -> Principal:
        return cls(
            user_id=user.id,
            username=user.username,
            role=user.role,
            department_id=user.department_id,
            display_name=user.display_name,
            session_id=session_id,
        )

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    @property
    def can_write(self) -> bool:
        return self.role in {Role.ADMIN, Role.OFFICER}


# A system actor for scheduled jobs and email intake. Job-written rows have
# a NULL actor_id in inquiry_events, which is how the timeline distinguishes
# "the SLA sweep escalated this" from "someone escalated this".
SYSTEM = Principal(
    user_id=0, username="system", role=Role.ADMIN, department_id=None,
    display_name="UniFlow",
)


def visible_department_ids(session, principal: Principal) -> list[int] | None:
    """Departments this principal may read.

    None means "no restriction" (admin). Otherwise the officer's own
    department plus Triage -- Triage is shared because an unrouted inquiry
    has no owning department yet and anyone may claim it.
    """
    if principal.is_admin:
        return None

    triage_id = session.query(Department.id).filter(Department.is_triage.is_(True)).scalar()
    visible = [department_id for department_id in (triage_id, principal.department_id)
               if department_id is not None]
    return visible


def can_read(session, principal: Principal, inquiry: Inquiry) -> bool:
    scope = visible_department_ids(session, principal)
    if scope is None:
        return True
    # An inquiry with no department yet is unplaced, and therefore in the
    # shared Triage pool by definition.
    if inquiry.department_id is None:
        return True
    return inquiry.department_id in scope


def assert_can_read(session, principal: Principal, inquiry: Inquiry) -> None:
    if not can_read(session, principal, inquiry):
        # 404 rather than 403 would hide existence, but officers are
        # trusted staff and a clear message is more useful than
        # obscuring that another department's queue exists.
        raise PermissionDeniedError(
            f"{principal.username} may not view inquiries outside their department"
        )


def assert_can_act_on(session, principal: Principal, inquiry: Inquiry) -> None:
    """The check every mutating service method makes before writing."""
    if not principal.can_write:
        raise PermissionDeniedError(f"{principal.username} has read-only access")
    assert_can_read(session, principal, inquiry)


def assert_admin(principal: Principal) -> None:
    if not principal.is_admin:
        raise PermissionDeniedError("This action requires an administrator")
