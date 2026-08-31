"""
workflow.py
-----------
Moving a ticket through its lifecycle: status, assignment, routing out of
Triage, escalation.

Three defects in the old implementation are fixed here.

Lost updates. update_queue_item read queue_status, first_response_at and
resolved_at, computed the new values in Python, then wrote them back --
with no transaction and no guard. Two officers acting at the same time
silently overwrote each other. Every mutation here is a single UPDATE
guarded by the row's version, so the second writer gets a 409 instead.

Split transactions. assign_department wrote the department, committed,
closed the connection, and only then opened a *second* connection to log
the review event. A crash in between left the audit log disagreeing with
the ticket. It is now one transaction.

Silent authorization. Both functions took a raw inquiry id and mutated it
with no ownership check whatsoever; can_access() existed but was never
called. Every entry point here calls assert_can_act_on first.
"""

from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa

from uniflow.core.enums import EventType, QueueStatus, ReviewAction, RoutingDecision
from uniflow.core.errors import ConflictError, NotFoundError, ValidationError
from uniflow.db.models import Department, Inquiry, InquiryEvent, ReviewEvent, User
from uniflow.domain import sla
from uniflow.domain.principal import Principal, assert_can_act_on, assert_can_read


def get(session, principal: Principal, inquiry_id: int) -> Inquiry:
    inquiry = session.get(Inquiry, inquiry_id)
    if inquiry is None:
        raise NotFoundError(f"Inquiry {inquiry_id} does not exist")
    assert_can_read(session, principal, inquiry)
    return inquiry


def _bump(session, inquiry: Inquiry, expected_version: int | None) -> None:
    """Claims the row for this writer, or refuses.

    The UPDATE ... WHERE id = :id AND version = :version is the whole
    concurrency control: if another writer got there first the row count
    is zero and nothing was changed.
    """
    if expected_version is None:
        expected_version = inquiry.version

    result = session.execute(
        sa.update(Inquiry)
        .where(Inquiry.id == inquiry.id, Inquiry.version == expected_version)
        .values(version=expected_version + 1, updated_at=datetime.now(timezone.utc))
    )
    if result.rowcount == 0:
        raise ConflictError(
            f"Inquiry {inquiry.reference} was modified by someone else. "
            "Reload and try again.",
            detail={"expected_version": expected_version, "actual_version": inquiry.version},
        )
    inquiry.version = expected_version + 1


def update_status(session, principal: Principal, inquiry_id: int, *,
                  status: QueueStatus | None = None,
                  assignee_id: int | None = ...,
                  expected_version: int | None = None) -> Inquiry:
    """Changes status and/or assignee.

    Stamps first_response_at the first time a ticket leaves 'New' -- that
    timestamp is what SLA compliance is measured against -- and
    resolved_at when it reaches a terminal state.

    assignee_id defaults to the ellipsis sentinel rather than None so that
    "leave the assignee alone" and "unassign this ticket" remain distinct.
    """
    inquiry = session.get(Inquiry, inquiry_id)
    if inquiry is None:
        raise NotFoundError(f"Inquiry {inquiry_id} does not exist")
    assert_can_act_on(session, principal, inquiry)

    _bump(session, inquiry, expected_version)
    now = datetime.now(timezone.utc)
    previous_status = inquiry.status

    if assignee_id is not ...:
        if assignee_id is not None:
            assignee = session.get(User, assignee_id)
            if assignee is None or not assignee.is_active:
                raise ValidationError(f"User {assignee_id} cannot be assigned work")
        previous_assignee = inquiry.assignee_id
        inquiry.assignee_id = assignee_id
        if previous_assignee != assignee_id:
            session.add(InquiryEvent(
                inquiry_id=inquiry.id, actor_id=principal.user_id or None,
                event_type=EventType.ASSIGNED, created_at=now,
                from_value={"assignee_id": previous_assignee},
                to_value={"assignee_id": assignee_id},
            ))

    if status is not None and status != previous_status:
        inquiry.status = status

        if inquiry.first_response_at is None and status is not QueueStatus.NEW:
            inquiry.first_response_at = now
            session.add(InquiryEvent(
                inquiry_id=inquiry.id, actor_id=principal.user_id or None,
                event_type=EventType.RESPONDED, created_at=now,
                to_value={"first_response_at": now.isoformat()},
            ))

        if status.is_terminal and inquiry.resolved_at is None:
            inquiry.resolved_at = now
            session.add(InquiryEvent(
                inquiry_id=inquiry.id, actor_id=principal.user_id or None,
                event_type=EventType.RESOLVED, created_at=now,
                to_value={"resolved_at": now.isoformat(),
                          "sla_state": sla.sla_state(inquiry, now).value},
            ))
        elif not status.is_terminal and inquiry.resolved_at is not None:
            # Reopening. The old code cleared resolved_at on any non-Resolved
            # status without recording that it had happened, so a ticket
            # could silently lose its resolution timestamp.
            inquiry.resolved_at = None
            session.add(InquiryEvent(
                inquiry_id=inquiry.id, actor_id=principal.user_id or None,
                event_type=EventType.REOPENED, created_at=now,
                from_value={"status": previous_status.value},
                to_value={"status": status.value},
            ))

        session.add(InquiryEvent(
            inquiry_id=inquiry.id, actor_id=principal.user_id or None,
            event_type=EventType.STATUS_CHANGED, created_at=now,
            from_value={"status": previous_status.value},
            to_value={"status": status.value},
        ))

    session.flush()
    return inquiry


def route(session, principal: Principal, inquiry_id: int, *, department_id: int,
          expected_version: int | None = None, note: str | None = None) -> Inquiry:
    """Moves an inquiry out of Triage into a real department queue.

    Recorded as a review event, because a human placing an abstained
    inquiry is exactly the supervision signal retraining needs -- and it
    is the labelled data the selective-prediction design is meant to
    generate. Whether that counts as an acceptance or a correction is
    decided against the *prediction*, not against the ticket's current
    department, which is what the old code compared and got wrong for any
    ticket that had already been moved once.
    """
    inquiry = session.get(Inquiry, inquiry_id)
    if inquiry is None:
        raise NotFoundError(f"Inquiry {inquiry_id} does not exist")
    assert_can_act_on(session, principal, inquiry)

    department = session.get(Department, department_id)
    if department is None or not department.is_active:
        raise ValidationError(f"Department {department_id} is not available")
    if department.is_triage:
        raise ValidationError("Use the triage queue only as a source, not a destination")

    _bump(session, inquiry, expected_version)
    now = datetime.now(timezone.utc)
    previous = inquiry.department

    prediction = inquiry.current_prediction
    predicted_label = prediction.category_label if prediction else None
    action = (
        ReviewAction.ACCEPT if predicted_label == department.code else ReviewAction.CORRECT
    )

    inquiry.department_id = department.id
    inquiry.routing = RoutingDecision.AUTO
    # The deadline was set from the Triage queue's policy; a department may
    # have negotiated its own, so it is recomputed from the original
    # creation time rather than from now -- the student's clock started
    # when they wrote in, not when we finished deciding.
    inquiry.sla_due_at = sla.due_at(
        session, inquiry.urgency, department.id, created_at=inquiry.created_at
    )

    session.add(ReviewEvent(
        inquiry_id=inquiry.id,
        prediction_id=prediction.id if prediction else None,
        reviewer_id=principal.user_id,
        action=action,
        corrected_department_id=department.id,
        corrected_urgency=inquiry.urgency,
        note=note,
        created_at=now,
    ))
    session.add(InquiryEvent(
        inquiry_id=inquiry.id, actor_id=principal.user_id or None,
        event_type=EventType.ROUTED, created_at=now,
        from_value={"department": previous.code if previous else None,
                    "routing": RoutingDecision.TRIAGE.value},
        to_value={"department": department.code, "routing": RoutingDecision.AUTO.value,
                  "predicted": predicted_label, "action": action.value},
    ))

    session.flush()
    return inquiry


def escalate(session, inquiry_ids: list[int], *, actor_id: int | None = None) -> int:
    """Flags breached tickets. Called by the SLA sweep, not by a page render.

    Set-based and idempotent: the WHERE clause re-checks escalated_at IS
    NULL, so two overlapping sweeps cannot double-escalate and cannot
    write duplicate timeline entries.
    """
    if not inquiry_ids:
        return 0
    now = datetime.now(timezone.utc)

    result = session.execute(
        sa.update(Inquiry)
        .where(Inquiry.id.in_(inquiry_ids), Inquiry.escalated_at.is_(None))
        .values(escalated_at=now, updated_at=now, version=Inquiry.version + 1)
    )
    if result.rowcount:
        escalated = session.execute(
            sa.select(Inquiry.id, Inquiry.reference, Inquiry.sla_due_at)
            .where(Inquiry.id.in_(inquiry_ids), Inquiry.escalated_at == now)
        ).all()
        for row in escalated:
            session.add(InquiryEvent(
                inquiry_id=row.id, actor_id=actor_id,
                event_type=EventType.ESCALATED, created_at=now,
                to_value={"reason": "sla_breach",
                          "sla_due_at": row.sla_due_at.isoformat() if row.sla_due_at else None},
            ))
    session.flush()
    return result.rowcount
