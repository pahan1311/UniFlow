"""
review.py
---------
The human-in-the-loop step: an officer accepts or corrects a prediction.

This is what closes the loop: an incremental retraining script would
consume these events to build a training set, weighting corrections more
heavily than acceptances, and refuse to promote unless macro-F1 improved
on the frozen test split. This project does not currently include that
script; the schema and the unconsumed()/mark_consumed() functions below
are what it would build on.

Two changes from the old log_review. The event now records which
*prediction* was being judged, not just which inquiry -- without that a
correction cannot be attributed to a model version once a promotion has
happened, which is precisely when you most want to know. And the review
no longer mirrors corrected_category / corrected_urgency back onto the
inquiry as a second source of truth; the ticket's department and urgency
are the current state, the review events are the history, and neither
pretends to be the other.
"""

from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa

from uniflow.core.enums import EventType, ReviewAction, RoutingDecision, Urgency
from uniflow.core.errors import NotFoundError, ValidationError
from uniflow.db.models import Department, Inquiry, InquiryEvent, ReviewEvent, TrainingRun
from uniflow.domain import sla
from uniflow.domain.principal import Principal, assert_can_act_on


def record(session, principal: Principal, inquiry_id: int, *,
           action: ReviewAction,
           corrected_department_id: int | None = None,
           corrected_urgency: Urgency | None = None,
           note: str | None = None,
           expected_version: int | None = None) -> ReviewEvent:
    """Logs one accept or correct, and applies the correction to the ticket."""
    from uniflow.domain.workflow import _bump  # same-layer helper, kept private

    inquiry = session.get(Inquiry, inquiry_id)
    if inquiry is None:
        raise NotFoundError(f"Inquiry {inquiry_id} does not exist")
    assert_can_act_on(session, principal, inquiry)

    if action is ReviewAction.CORRECT and not (corrected_department_id or corrected_urgency):
        raise ValidationError("A correction must change the department, the urgency, or both")

    _bump(session, inquiry, expected_version)
    now = datetime.now(timezone.utc)
    prediction = inquiry.current_prediction

    changes: dict = {}

    if corrected_department_id is not None:
        department = session.get(Department, corrected_department_id)
        if department is None or not department.is_active:
            raise ValidationError(f"Department {corrected_department_id} is not available")
        if inquiry.department_id != department.id:
            changes["department"] = {
                "from": inquiry.department.code if inquiry.department else None,
                "to": department.code,
            }
            inquiry.department_id = department.id
            # A human has placed it, so it is no longer an abstention.
            inquiry.routing = RoutingDecision.AUTO

    if corrected_urgency is not None and inquiry.urgency != corrected_urgency:
        changes["urgency"] = {
            "from": inquiry.urgency.value if inquiry.urgency else None,
            "to": corrected_urgency.value,
        }
        inquiry.urgency = corrected_urgency
        # Urgency drives the deadline, so correcting it has to move the
        # deadline too -- otherwise the correction is cosmetic. Measured
        # from creation, so a ticket corrected to High does not get a
        # fresh four hours it has already spent.
        inquiry.sla_due_at = sla.due_at(
            session, corrected_urgency, inquiry.department_id, created_at=inquiry.created_at
        )

    event = ReviewEvent(
        inquiry_id=inquiry.id,
        prediction_id=prediction.id if prediction else None,
        reviewer_id=principal.user_id,
        action=action,
        corrected_department_id=corrected_department_id,
        corrected_urgency=corrected_urgency,
        note=note,
        created_at=now,
    )
    session.add(event)
    session.add(InquiryEvent(
        inquiry_id=inquiry.id, actor_id=principal.user_id or None,
        event_type=EventType.STATUS_CHANGED if changes else EventType.ROUTED,
        created_at=now,
        to_value={"review": action.value, "changes": changes,
                  "predicted_category": prediction.category_label if prediction else None},
    ))
    session.flush()
    return event


# ---------------------------------------------------------------------------
# The retraining input set
# ---------------------------------------------------------------------------

def unconsumed(session, limit: int | None = None) -> list[ReviewEvent]:
    """Review events not yet folded into a retraining run, oldest first."""
    query = (
        sa.select(ReviewEvent)
        .where(ReviewEvent.consumed_by_run_id.is_(None))
        .order_by(ReviewEvent.created_at, ReviewEvent.id)
    )
    if limit:
        query = query.limit(limit)
    return list(session.execute(query).scalars())


def mark_consumed(session, event_ids: list[int], run: TrainingRun) -> int:
    """Attributes a set of review events to the run that absorbed them.

    consumed_by_training was a 0/1 flag, so once a correction had been
    used there was no way to tell which run used it or what that run
    produced.
    """
    if not event_ids:
        return 0
    result = session.execute(
        sa.update(ReviewEvent)
        .where(ReviewEvent.id.in_(event_ids), ReviewEvent.consumed_by_run_id.is_(None))
        .values(consumed_by_run_id=run.id)
    )
    session.flush()
    return result.rowcount


def correction_rate(session, window: int = 50) -> float | None:
    """Share of the last N reviews that were corrections, not acceptances.

    A rising correction rate is the cheapest available drift signal, so
    the analytics page plots it. Counted in SQL; the old version pulled
    the rows and counted them in Python.
    """
    recent = (
        sa.select(ReviewEvent.action)
        .order_by(ReviewEvent.id.desc())
        .limit(window)
        .subquery()
    )
    total, corrections = session.execute(
        sa.select(
            sa.func.count(),
            sa.func.sum(
                sa.case((recent.c.action == ReviewAction.CORRECT.value, 1), else_=0)
            ),
        ).select_from(recent)
    ).one()
    if not total:
        return None
    return float(corrections or 0) / float(total)
