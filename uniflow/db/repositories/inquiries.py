"""
inquiries.py
------------
Reading the queue.

fetch_all() was `SELECT * FROM inquiries ORDER BY id DESC` with no filter
and no LIMIT, run on every Streamlit rerun -- which is to say on every
widget interaction. The whole table went into pandas, and then filtering,
department scoping, SLA annotation and every aggregate were computed in
Python. That is O(table) per click, permanently.

Here filtering, scoping, sorting and pagination are one query, and the
department scope is a WHERE clause applied before any row leaves the
database rather than a pandas mask applied after they all have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.orm import selectinload

from uniflow.core.enums import QueueStatus, RoutingDecision, SlaState, Urgency
from uniflow.db.models import Inquiry
from uniflow.domain import sla
from uniflow.domain.principal import Principal, visible_department_ids

MAX_PAGE_SIZE = 200


@dataclass
class InquiryFilter:
    department_ids: list[int] = field(default_factory=list)
    statuses: list[QueueStatus] = field(default_factory=list)
    urgencies: list[Urgency] = field(default_factory=list)
    routings: list[RoutingDecision] = field(default_factory=list)
    sla_states: list[SlaState] = field(default_factory=list)
    assignee_id: int | None = None
    unassigned: bool = False
    escalated_only: bool = False
    open_only: bool = False
    duplicates_only: bool = False
    search: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None


@dataclass
class Page:
    items: list[Inquiry]
    total: int
    limit: int
    offset: int
    sla_states: dict[int, SlaState]

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


def _apply(query, filters: InquiryFilter, now: datetime):
    if filters.department_ids:
        query = query.where(Inquiry.department_id.in_(filters.department_ids))
    if filters.statuses:
        query = query.where(Inquiry.status.in_(filters.statuses))
    if filters.urgencies:
        query = query.where(Inquiry.urgency.in_(filters.urgencies))
    if filters.routings:
        query = query.where(Inquiry.routing.in_(filters.routings))
    if filters.assignee_id is not None:
        query = query.where(Inquiry.assignee_id == filters.assignee_id)
    if filters.unassigned:
        query = query.where(Inquiry.assignee_id.is_(None))
    if filters.escalated_only:
        query = query.where(Inquiry.escalated_at.isnot(None))
    if filters.open_only:
        query = query.where(Inquiry.resolved_at.is_(None))
    if filters.duplicates_only:
        query = query.where(Inquiry.duplicate_of_id.isnot(None))
    if filters.created_after:
        query = query.where(Inquiry.created_at >= filters.created_after)
    if filters.created_before:
        query = query.where(Inquiry.created_at < filters.created_before)
    if filters.search:
        # A LIKE scan is honest at this scale. On PostgreSQL the GIN
        # to_tsvector index in the migration makes the same call a full
        # text search; this is the one place that would change.
        pattern = f"%{filters.search.strip()}%"
        query = query.where(
            sa.or_(Inquiry.raw_text.ilike(pattern), Inquiry.reference.ilike(pattern))
        )
    if filters.sla_states:
        wanted = [state.value for state in filters.sla_states]
        query = query.where(sla.state_expression(now).in_(wanted))
    return query


def scope_for(session, principal: Principal, filters: InquiryFilter) -> InquiryFilter:
    """Intersects the requested departments with what the principal may see.

    An officer asking for another department's queue gets an empty result,
    not that department's tickets, and never sees them in memory.
    """
    allowed = visible_department_ids(session, principal)
    if allowed is None:
        return filters
    if filters.department_ids:
        filters.department_ids = [d for d in filters.department_ids if d in allowed]
        if not filters.department_ids:
            filters.department_ids = [-1]  # deliberately matches nothing
    else:
        filters.department_ids = list(allowed) or [-1]
    return filters


def list_inquiries(session, principal: Principal, filters: InquiryFilter | None = None,
                   *, limit: int = 50, offset: int = 0,
                   order: str = "created_desc", now: datetime | None = None) -> Page:
    now = now or sla.utcnow()
    filters = scope_for(session, principal, filters or InquiryFilter())
    limit = max(1, min(limit, MAX_PAGE_SIZE))

    base = _apply(sa.select(Inquiry), filters, now)

    total = session.execute(
        sa.select(sa.func.count()).select_from(base.subquery())
    ).scalar_one()

    # High before Medium before Low; ties broken by soonest deadline --
    # the priority a real work queue needs, computed in SQL.
    urgency_rank = sa.case(
        (Inquiry.urgency == Urgency.HIGH.value, 0),
        (Inquiry.urgency == Urgency.MEDIUM.value, 1),
        (Inquiry.urgency == Urgency.LOW.value, 2),
        else_=3,
    )

    ordering = {
        "created_desc": (Inquiry.created_at.desc(), Inquiry.id.desc()),
        "created_asc": (Inquiry.created_at.asc(), Inquiry.id.asc()),
        # Soonest deadline first, with unresolved work ahead of resolved.
        "due_asc": (Inquiry.sla_due_at.asc().nullslast(), Inquiry.id.desc()),
        "updated_desc": (Inquiry.updated_at.desc(), Inquiry.id.desc()),
        "priority": (urgency_rank, Inquiry.sla_due_at.asc().nullslast(), Inquiry.id.desc()),
    }.get(order, (Inquiry.created_at.desc(), Inquiry.id.desc()))

    rows = list(
        session.execute(
            base.options(selectinload(Inquiry.current_prediction))
            .order_by(*ordering)
            .limit(limit)
            .offset(offset)
        ).scalars()
    )

    # SLA state is time-dependent, so it is computed rather than stored.
    # Done here once per page of results, not per row per render.
    states = {row.id: sla.sla_state(row, now) for row in rows}
    return Page(items=rows, total=total, limit=limit, offset=offset, sla_states=states)


def breached_ids(session, now: datetime | None = None, limit: int = 500) -> list[int]:
    """Ids the SLA sweep should escalate. One indexed query, no rows loaded."""
    now = now or sla.utcnow()
    return list(
        session.execute(
            sa.select(Inquiry.id)
            .where(sla.breached_filter(now))
            .order_by(Inquiry.sla_due_at)
            .limit(limit)
        ).scalars()
    )


def timeline(session, inquiry_id: int):
    from uniflow.db.models import InquiryEvent

    return list(
        session.execute(
            sa.select(InquiryEvent)
            .where(InquiryEvent.inquiry_id == inquiry_id)
            .order_by(InquiryEvent.created_at, InquiryEvent.id)
        ).scalars()
    )
