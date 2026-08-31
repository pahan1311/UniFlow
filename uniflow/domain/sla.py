"""
sla.py
------
Turns the predicted urgency label into a measurable service commitment.

Without this module the urgency model is decorative: it prints a word
next to a ticket and nothing follows from it. Here each urgency level
carries a response-time target, every ticket gets a deadline at intake,
and breaches escalate.

That also gives the evaluation an end-to-end, operational metric --
SLA compliance rate -- which is stronger evidence than classification
accuracy alone, because it measures whether accurate urgency prediction
actually produced faster handling.

Two things changed from app/sla.py. The state functions are unchanged and
remain the reference implementation, but they now take timezone-aware UTC
datetimes rather than parsing naive local ISO strings. And annotate() /
find_escalations(), which walked a pandas DataFrame of the entire table on
every page render, are replaced by `state_expression()` -- the same logic
as a SQL CASE, so filtering and counting happen in the database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import sqlalchemy as sa

from uniflow.core.enums import SlaState, Urgency
from uniflow.db.models import Inquiry, SlaPolicy

STATE_LABEL = {
    SlaState.MET: "Met",
    SlaState.ON_TRACK: "On track",
    SlaState.AT_RISK: "At risk",
    SlaState.BREACHED: "Breached",
}

DEFAULT_WARNING_FRACTION = Decimal("0.75")

# Used only when the sla_policies table is empty, which should not happen
# after seeding but must not take intake down if it does.
FALLBACK_HOURS = {Urgency.HIGH: 4, Urgency.MEDIUM: 24, Urgency.LOW: 72}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------

def resolve_policy(session, urgency: Urgency, department_id: int | None,
                   at: datetime | None = None) -> tuple[Decimal, Decimal]:
    """The (response_hours, warning_fraction) in force for this urgency.

    A department-specific row wins over the global default, and among
    rows of equal specificity the newest effective_from wins. Resolving
    at intake rather than at read time means changing a target tomorrow
    does not retroactively rewrite today's deadlines.
    """
    at = at or utcnow()
    row = session.execute(
        sa.select(SlaPolicy)
        .where(
            SlaPolicy.urgency == urgency,
            SlaPolicy.effective_from <= at,
            sa.or_(
                SlaPolicy.department_id == department_id,
                SlaPolicy.department_id.is_(None),
            ),
        )
        # NULL department_id sorts last, so a specific policy is preferred.
        .order_by(
            SlaPolicy.department_id.is_(None),
            SlaPolicy.effective_from.desc(),
        )
        .limit(1)
    ).scalar_one_or_none()

    if row is None:
        return Decimal(FALLBACK_HOURS.get(urgency, 72)), DEFAULT_WARNING_FRACTION
    return row.response_hours, row.warning_fraction


def due_at(session, urgency: Urgency, department_id: int | None,
           created_at: datetime | None = None) -> datetime:
    """Response deadline implied by the predicted urgency label.

    This is the mechanism that converts an urgency *prediction* into an
    operational commitment the system can be evaluated against.
    """
    created_at = created_at or utcnow()
    hours, _ = resolve_policy(session, urgency, department_id, created_at)
    return created_at + timedelta(hours=float(hours))


# ---------------------------------------------------------------------------
# State, in Python (the reference implementation)
# ---------------------------------------------------------------------------

def sla_state(inquiry: Inquiry, now: datetime | None = None,
              warning_fraction: Decimal = DEFAULT_WARNING_FRACTION) -> SlaState:
    """Classifies one ticket's SLA position.

    A resolved ticket is judged on when it was resolved, not on the
    current time -- otherwise every old ticket would eventually read as
    breached and compliance statistics would be meaningless.
    """
    now = now or utcnow()
    due = inquiry.sla_due_at
    if due is None:
        return SlaState.ON_TRACK

    finished = inquiry.resolved_at
    if finished is not None:
        return SlaState.MET if finished <= due else SlaState.BREACHED

    if now > due:
        return SlaState.BREACHED

    created = inquiry.created_at or now
    window = (due - created).total_seconds()
    if window <= 0:
        return SlaState.AT_RISK
    elapsed_fraction = (now - created).total_seconds() / window
    return SlaState.AT_RISK if elapsed_fraction >= float(warning_fraction) else SlaState.ON_TRACK


def time_remaining(inquiry: Inquiry, now: datetime | None = None) -> str:
    """Human-readable countdown, or overdue amount, for the ticket card."""
    now = now or utcnow()
    due = inquiry.sla_due_at
    if due is None:
        return "No target"

    finished = inquiry.resolved_at
    if finished is not None:
        delta = due - finished
        prefix = "Resolved with" if delta.total_seconds() >= 0 else "Resolved late by"
        return f"{prefix} {humanise(abs(delta.total_seconds()))}" + (
            " to spare" if delta.total_seconds() >= 0 else ""
        )

    delta = (due - now).total_seconds()
    if delta < 0:
        return f"Overdue by {humanise(abs(delta))}"
    return f"{humanise(delta)} remaining"


def humanise(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


# ---------------------------------------------------------------------------
# State, in SQL (what the queries actually use)
# ---------------------------------------------------------------------------

def state_expression(now: datetime | None = None,
                     warning_fraction: float = float(DEFAULT_WARNING_FRACTION)):
    """The same four-way classification as sla_state(), as a SQL CASE.

    Lets the queue and analytics queries filter and GROUP BY on SLA state
    instead of loading the table into pandas and calling df.apply. The
    test suite asserts this expression and sla_state() agree row for row;
    if they ever diverge, that test is the thing that catches it.

    The elapsed-fraction comparison is rearranged to avoid dividing by an
    interval, which SQLite and PostgreSQL spell differently:

        (now - created) / (due - created) >= f   becomes
        now >= created + f * (due - created)

    Both sides are then epoch seconds, which is plain arithmetic on either
    engine: EXTRACT(EPOCH FROM col) on PostgreSQL, STRFTIME('%s', col) on
    SQLite, and SQLAlchemy emits whichever applies. `now` is folded to a
    literal in Python so no date function is applied to a bind parameter.
    """
    now = now or utcnow()
    created = sa.extract("epoch", Inquiry.created_at)
    due = sa.extract("epoch", Inquiry.sla_due_at)
    now_epoch = sa.literal(now.timestamp())

    warn_point = created + warning_fraction * (due - created)

    return sa.case(
        (Inquiry.sla_due_at.is_(None), sa.literal(SlaState.ON_TRACK.value)),
        (
            sa.and_(Inquiry.resolved_at.isnot(None), Inquiry.resolved_at <= Inquiry.sla_due_at),
            sa.literal(SlaState.MET.value),
        ),
        (Inquiry.resolved_at.isnot(None), sa.literal(SlaState.BREACHED.value)),
        (Inquiry.sla_due_at < now, sa.literal(SlaState.BREACHED.value)),
        (now_epoch >= warn_point, sa.literal(SlaState.AT_RISK.value)),
        else_=sa.literal(SlaState.ON_TRACK.value),
    )


def breached_filter(now: datetime | None = None):
    """Unresolved tickets past their deadline and not yet escalated.

    This is the SLA sweep's entire WHERE clause. Escalation used to happen
    inside the render path -- find_escalations() over a DataFrame, then a
    write per id -- so nothing escalated unless somebody opened the
    dashboard.
    """
    now = now or utcnow()
    return sa.and_(
        Inquiry.resolved_at.is_(None),
        Inquiry.escalated_at.is_(None),
        Inquiry.sla_due_at.isnot(None),
        Inquiry.sla_due_at < now,
    )
