"""
inquiries.py
------------
The queue: intake, listing, workflow, review, explanation.

Every list endpoint is paginated and pushes filtering into SQL. An
officer's GET /inquiries emits `WHERE department_id IN (own, triage)` --
the authorization is a predicate applied before rows leave the database,
where the old design read every department's tickets into memory and then
hid the ones the officer was not allowed to see.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from uniflow.api import presenters
from uniflow.api.deps import PrincipalDep, SessionDep
from uniflow.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    DuplicateOut,
    EventOut,
    ExplanationOut,
    InquiryCreate,
    InquiryDetail,
    InquiryPage,
    IntakeResponse,
    ReviewOut,
    ReviewRequest,
    RouteRequest,
    StatusUpdate,
)
from uniflow.core.enums import IntakeSource, QueueStatus, RoutingDecision, SlaState, Urgency
from uniflow.core.errors import NotFoundError
from uniflow.db.repositories import inquiries as repo
from uniflow.domain import explanations, policy, review, triage, workflow
from uniflow.ml import registry

router = APIRouter(tags=["inquiries"])


@router.post("/inquiries", response_model=IntakeResponse, status_code=201)
def create(payload: InquiryCreate, session: SessionDep, principal: PrincipalDep):
    """Submits one inquiry. The same service the email poller calls."""
    result = triage.submit(
        session,
        raw_text=payload.text,
        principal=principal,
        # A client cannot claim to be the email channel: that would let it
        # forge a source_ref and bypass the intake idempotency guard.
        source=IntakeSource.API if payload.source is IntakeSource.EMAIL else payload.source,
        student_email=payload.student_email,
        student_ref=payload.student_ref,
    )
    return IntakeResponse(
        inquiry=presenters.inquiry_detail(result.inquiry, routing_reason=result.routing_reason),
        prediction=presenters.prediction_out(result.prediction),
        duplicate=presenters.duplicate_out(result.duplicate),
        routing_reason=result.routing_reason,
    )


@router.get("/inquiries", response_model=InquiryPage)
def list_inquiries(
    session: SessionDep,
    principal: PrincipalDep,
    department_id: Annotated[list[int] | None, Query()] = None,
    status: Annotated[list[QueueStatus] | None, Query()] = None,
    urgency: Annotated[list[Urgency] | None, Query()] = None,
    routing: Annotated[list[RoutingDecision] | None, Query()] = None,
    sla_state: Annotated[list[SlaState] | None, Query()] = None,
    assignee_id: int | None = None,
    unassigned: bool = False,
    escalated_only: bool = False,
    open_only: bool = False,
    duplicates_only: bool = False,
    search: str | None = None,
    order: str = "created_desc",
    limit: int = Query(default=50, ge=1, le=repo.MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
):
    filters = repo.InquiryFilter(
        department_ids=department_id or [],
        statuses=status or [],
        urgencies=urgency or [],
        routings=routing or [],
        sla_states=sla_state or [],
        assignee_id=assignee_id,
        unassigned=unassigned,
        escalated_only=escalated_only,
        open_only=open_only,
        duplicates_only=duplicates_only,
        search=search,
    )
    page = repo.list_inquiries(
        session, principal, filters, limit=limit, offset=offset, order=order
    )
    return InquiryPage(
        items=[
            presenters.inquiry_out(item, sla_state=page.sla_states.get(item.id))
            for item in page.items
        ],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        has_more=page.has_more,
    )


@router.get("/inquiries/{inquiry_id}", response_model=InquiryDetail)
def get(inquiry_id: int, session: SessionDep, principal: PrincipalDep, response: Response):
    inquiry = workflow.get(session, principal, inquiry_id)
    # ETag carries the optimistic-lock version, so a client can send it
    # straight back as If-Match on the next write.
    response.headers["ETag"] = f'W/"{inquiry.version}"'
    return presenters.inquiry_detail(inquiry)


@router.patch("/inquiries/{inquiry_id}", response_model=InquiryDetail)
def update(inquiry_id: int, payload: StatusUpdate, session: SessionDep,
           principal: PrincipalDep):
    """Changes status and/or assignee.

    Send expected_version to make the write conditional. A mismatch is a
    409 rather than a silent overwrite of whatever the other officer did.
    """
    assignee: object = ...
    if payload.clear_assignee:
        assignee = None
    elif payload.assignee_id is not None:
        assignee = payload.assignee_id

    inquiry = workflow.update_status(
        session, principal, inquiry_id,
        status=payload.status,
        assignee_id=assignee,
        expected_version=payload.expected_version,
    )
    return presenters.inquiry_detail(inquiry)


@router.post("/inquiries/{inquiry_id}/route", response_model=InquiryDetail)
def route(inquiry_id: int, payload: RouteRequest, session: SessionDep,
          principal: PrincipalDep):
    """Places an abstained inquiry into a department queue."""
    inquiry = workflow.route(
        session, principal, inquiry_id,
        department_id=payload.department_id,
        note=payload.note,
        expected_version=payload.expected_version,
    )
    return presenters.inquiry_detail(inquiry)


@router.post("/inquiries/{inquiry_id}/review", response_model=ReviewOut, status_code=201)
def submit_review(inquiry_id: int, payload: ReviewRequest, session: SessionDep,
                  principal: PrincipalDep):
    event = review.record(
        session, principal, inquiry_id,
        action=payload.action,
        corrected_department_id=payload.corrected_department_id,
        corrected_urgency=payload.corrected_urgency,
        note=payload.note,
        expected_version=payload.expected_version,
    )
    return ReviewOut.model_validate(event)


@router.get("/inquiries/{inquiry_id}/explanation", response_model=ExplanationOut)
def explanation(inquiry_id: int, session: SessionDep, principal: PrincipalDep,
                task: str = Query(default="category", pattern="^(category|urgency)$")):
    """Integrated Gradients attributions, computed once and cached."""
    return presenters.explanation_out(
        explanations.get_or_compute(session, principal, inquiry_id, task=task)
    )


@router.get("/inquiries/{inquiry_id}/duplicates", response_model=DuplicateOut | None)
def duplicate(inquiry_id: int, session: SessionDep, principal: PrincipalDep):
    inquiry = workflow.get(session, principal, inquiry_id)
    if inquiry.duplicate_of_id is None:
        return None
    from uniflow.db.models import Inquiry

    other = session.get(Inquiry, inquiry.duplicate_of_id)
    if other is None:
        raise NotFoundError("The linked duplicate no longer exists")
    return DuplicateOut(
        inquiry_id=other.id,
        reference=other.reference,
        similarity=float(inquiry.duplicate_score or 0),
    )


@router.get("/inquiries/{inquiry_id}/events", response_model=list[EventOut])
def events(inquiry_id: int, session: SessionDep, principal: PrincipalDep):
    """The ticket timeline, which the old schema had nowhere to record."""
    workflow.get(session, principal, inquiry_id)  # authorization
    return [EventOut.model_validate(event) for event in repo.timeline(session, inquiry_id)]


@router.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest, session: SessionDep, principal: PrincipalDep):
    """Runs the model without creating anything.

    For benchmarking and for callers that want a routing opinion without
    filing a ticket.
    """
    from uniflow.ml.inference import classify_inquiry

    active = policy.active_policy(session)
    model_version = policy.active_model_version(session, "category")
    result = classify_inquiry(
        payload.text,
        registry.get(),
        thresholds=active.as_thresholds(),
        calibration=active.as_calibration(),
    )
    return ClassifyResponse(
        category_label=result["category_label"],
        category_confidence=result["category_confidence"],
        category_energy=result["category_energy"],
        category_probs=result["category_probs"],
        urgency_label=result["urgency_label"],
        urgency_confidence=result["urgency_confidence"],
        urgency_probs=result["urgency_probs"],
        routing_decision=result["routing_decision"],
        routing_reason=result["routing_reason"],
        latency_ms=result.get("latency_ms"),
        policy_version=active.version,
        model_version=f"{model_version.name}:{model_version.version}",
    )
