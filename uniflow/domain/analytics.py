"""
analytics.py
------------
Aggregates, computed in SQL.

There was not a single GROUP BY anywhere in the old codebase. Every
figure on the analytics page -- per-department volume, SLA compliance,
routing-decision counts, mean confidence, volume over time -- was derived
by reading the entire inquiries table into pandas and grouping it there,
on every page load.

Each function below returns rows the UI can plot directly. The database
returns tens of rows instead of the whole table.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import sqlalchemy as sa

from uniflow.core.enums import QueueStatus, RoutingDecision, SlaState
from uniflow.db.models import Department, Inquiry, Prediction, ReviewEvent
from uniflow.domain import review, sla
from uniflow.domain.principal import Principal, visible_department_ids


def _scoped(query, session, principal: Principal):
    allowed = visible_department_ids(session, principal)
    if allowed is None:
        return query
    return query.where(Inquiry.department_id.in_(allowed or [-1]))


def overview(session, principal: Principal, now: datetime | None = None) -> dict:
    """The headline counters, in one round trip rather than one per tile."""
    now = now or sla.utcnow()
    state = sla.state_expression(now)

    row = session.execute(
        _scoped(
            sa.select(
                sa.func.count().label("total"),
                sa.func.sum(sa.case((Inquiry.resolved_at.is_(None), 1), else_=0)).label("open"),
                sa.func.sum(
                    sa.case((Inquiry.routing != RoutingDecision.AUTO.value, 1), else_=0)
                ).label("abstained"),
                sa.func.sum(sa.case((Inquiry.escalated_at.isnot(None), 1), else_=0))
                .label("escalated"),
                sa.func.sum(sa.case((Inquiry.duplicate_of_id.isnot(None), 1), else_=0))
                .label("duplicates"),
                sa.func.sum(sa.case((state == SlaState.BREACHED.value, 1), else_=0))
                .label("breached"),
                sa.func.sum(sa.case((state == SlaState.AT_RISK.value, 1), else_=0))
                .label("at_risk"),
            ),
            session,
            principal,
        )
    ).one()

    total = row.total or 0
    return {
        "total": total,
        "open": row.open or 0,
        "abstained": row.abstained or 0,
        "escalated": row.escalated or 0,
        "duplicates": row.duplicates or 0,
        "breached": row.breached or 0,
        "at_risk": row.at_risk or 0,
        # Coverage: the share auto-routed without a human. The other half
        # of the risk-coverage trade-off the thesis argues for.
        "coverage": (1 - (row.abstained or 0) / total) if total else None,
        "correction_rate": review.correction_rate(session),
    }


def by_department(session, principal: Principal, now: datetime | None = None) -> list[dict]:
    now = now or sla.utcnow()
    state = sla.state_expression(now)

    rows = session.execute(
        _scoped(
            sa.select(
                Department.code,
                Department.name,
                sa.func.count(Inquiry.id).label("total"),
                sa.func.sum(sa.case((Inquiry.resolved_at.is_(None), 1), else_=0)).label("open"),
                sa.func.sum(sa.case((state == SlaState.BREACHED.value, 1), else_=0))
                .label("breached"),
                sa.func.avg(Prediction.category_confidence).label("mean_confidence"),
            )
            .select_from(Inquiry)
            .join(Department, Department.id == Inquiry.department_id)
            .outerjoin(Prediction, Prediction.id == Inquiry.current_prediction_id),
            session,
            principal,
        )
        .group_by(Department.code, Department.name)
        .order_by(sa.func.count(Inquiry.id).desc())
    ).all()

    return [
        {
            "code": row.code,
            "name": row.name,
            "total": row.total,
            "open": row.open or 0,
            "breached": row.breached or 0,
            "mean_confidence": float(row.mean_confidence) if row.mean_confidence else None,
        }
        for row in rows
    ]


def sla_compliance(session, principal: Principal, now: datetime | None = None) -> dict:
    """Compliance measured over closed tickets only.

    An open ticket has not yet succeeded or failed, so counting it either
    way makes the rate meaningless. Median resolution time is computed
    from epoch differences so it works on both engines.
    """
    now = now or sla.utcnow()
    state = sla.state_expression(now)

    closed = session.execute(
        _scoped(
            sa.select(
                sa.func.count().label("closed"),
                sa.func.sum(sa.case((state == SlaState.MET.value, 1), else_=0)).label("met"),
                sa.func.avg(
                    sa.extract("epoch", Inquiry.resolved_at)
                    - sa.extract("epoch", Inquiry.created_at)
                ).label("mean_seconds"),
            ).where(Inquiry.resolved_at.isnot(None)),
            session,
            principal,
        )
    ).one()

    live = session.execute(
        _scoped(
            sa.select(
                state.label("state"), sa.func.count().label("count")
            ).where(Inquiry.resolved_at.is_(None)),
            session,
            principal,
        ).group_by(state)
    ).all()

    total_closed = closed.closed or 0
    return {
        "closed": total_closed,
        "met": closed.met or 0,
        "compliance_rate": (closed.met or 0) / total_closed if total_closed else None,
        "mean_resolution_hours": (
            float(closed.mean_seconds) / 3600 if closed.mean_seconds else None
        ),
        "open_by_state": {row.state: row.count for row in live},
    }


def volume_over_time(session, principal: Principal, days: int = 30,
                     now: datetime | None = None) -> list[dict]:
    """Daily intake, split by whether the system auto-routed or abstained."""
    now = now or sla.utcnow()
    since = now - timedelta(days=days)
    # date() is one of the few date functions spelled identically by both
    # SQLite and PostgreSQL when applied to a timestamp column.
    day = sa.func.date(Inquiry.created_at)

    rows = session.execute(
        _scoped(
            sa.select(
                day.label("day"),
                sa.func.count().label("total"),
                sa.func.sum(
                    sa.case((Inquiry.routing == RoutingDecision.AUTO.value, 1), else_=0)
                ).label("auto"),
                sa.func.sum(
                    sa.case((Inquiry.routing == RoutingDecision.TRIAGE.value, 1), else_=0)
                ).label("triage"),
                sa.func.sum(
                    sa.case(
                        (Inquiry.routing == RoutingDecision.OUT_OF_SCOPE.value, 1), else_=0
                    )
                ).label("out_of_scope"),
            ).where(Inquiry.created_at >= since),
            session,
            principal,
        )
        .group_by(day)
        .order_by(day)
    ).all()

    return [
        {
            "day": str(row.day),
            "total": row.total,
            "auto": row.auto or 0,
            "triage": row.triage or 0,
            "out_of_scope": row.out_of_scope or 0,
        }
        for row in rows
    ]


def confidence_histogram(session, principal: Principal, buckets: int = 10) -> list[dict]:
    """Calibrated confidence distribution, split by whether we abstained.

    This is the live-traffic view of the risk-coverage trade-off: the mass
    sitting just below tau is exactly what a threshold change would move.
    Reading it from stored predictions rather than the frozen test split
    is only possible because predictions is now a table.
    """
    width = 1.0 / buckets
    raw_bucket = sa.cast(Prediction.category_confidence / width, sa.Integer)
    # A confidence of exactly 1.0 would land one bucket past the end.
    # Clamped with CASE rather than min()/LEAST, which the two engines
    # spell differently.
    bucket = sa.case((raw_bucket >= buckets, buckets - 1), else_=raw_bucket)

    rows = session.execute(
        _scoped(
            sa.select(
                bucket.label("bucket"),
                Prediction.abstained,
                sa.func.count().label("count"),
            )
            .select_from(Prediction)
            .join(Inquiry, Inquiry.id == Prediction.inquiry_id),
            session,
            principal,
        )
        .group_by(bucket, Prediction.abstained)
        .order_by(bucket)
    ).all()

    return [
        {
            "lower": round(row.bucket * width, 3),
            "upper": round((row.bucket + 1) * width, 3),
            "abstained": bool(row.abstained),
            "count": row.count,
        }
        for row in rows
    ]


def routing_mix(session, principal: Principal) -> list[dict]:
    rows = session.execute(
        _scoped(
            sa.select(Inquiry.routing, sa.func.count().label("count")), session, principal
        ).group_by(Inquiry.routing)
    ).all()
    return [{"routing": row.routing.value, "count": row.count} for row in rows]


def review_activity(session, days: int = 30, now: datetime | None = None) -> list[dict]:
    """Accepts vs corrections per day -- the drift signal, as a time series."""
    now = now or sla.utcnow()
    since = now - timedelta(days=days)
    day = sa.func.date(ReviewEvent.created_at)

    rows = session.execute(
        sa.select(day.label("day"), ReviewEvent.action, sa.func.count().label("count"))
        .where(ReviewEvent.created_at >= since)
        .group_by(day, ReviewEvent.action)
        .order_by(day)
    ).all()
    return [
        {"day": str(row.day), "action": row.action.value, "count": row.count} for row in rows
    ]
