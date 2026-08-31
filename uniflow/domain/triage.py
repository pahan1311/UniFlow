"""
triage.py
---------
Intake: the single write path for a new inquiry.

Every channel goes through submit(): the web form, the IMAP poller, and
any future API client or student portal. Previously there were two
implementations of this flow -- app.py lines 529-545 and
ingest_email.process_message -- which were already subtly different and
would have kept drifting.

The whole of submit() is one transaction. The old sequence was not: it
classified, wrote the inquiry, then sent the acknowledgement by calling
smtplib inline, so a failure part-way through left an inquiry with no
acknowledgement and no record that one was owed.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import sqlalchemy as sa

from uniflow.core.enums import (
    TRIAGE_CODE,
    EventType,
    IntakeSource,
    QueueStatus,
    RoutingDecision,
    Urgency,
)
from uniflow.core.errors import DuplicateIntakeError, ValidationError
from uniflow.core.settings import get_settings
from uniflow.db.models import (
    Department,
    Inquiry,
    InquiryEmbedding,
    InquiryEvent,
    OutboundMessage,
    Prediction,
)
from uniflow.domain import dedupe, policy, sla
from uniflow.domain.principal import SYSTEM, Principal
from uniflow.ml import registry

logger = logging.getLogger(__name__)


@dataclass
class IntakeResult:
    inquiry: Inquiry
    prediction: Prediction
    duplicate: dedupe.DuplicateMatch | None
    routing_reason: str
    skipped: bool = False


def text_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _next_reference(session, inquiry_id: int, created_at: datetime) -> str:
    """Human-facing identifier: INQ-2026-000123.

    Derived from the surrogate key after flush, so it is unique without a
    second sequence and without a read-modify-write on a counter row.
    """
    return f"INQ-{created_at.year}-{inquiry_id:06d}"


def _department_for(session, label: str, decision: RoutingDecision) -> Department | None:
    """Maps a predicted label to a real department row.

    An abstention or an out-of-scope verdict goes to Triage. A confident
    prediction whose label has no matching department is a configuration
    error -- the model and the departments table disagree -- and is sent
    to Triage rather than dropped, because losing a student's inquiry is
    the worse failure.
    """
    if decision is not RoutingDecision.AUTO:
        return session.execute(
            sa.select(Department).where(Department.is_triage.is_(True))
        ).scalar_one_or_none()

    department = session.execute(
        sa.select(Department).where(Department.code == label, Department.is_active.is_(True))
    ).scalar_one_or_none()
    if department is None:
        logger.warning(
            "model predicted '%s' but no active department has that code; routing to Triage",
            label,
        )
        return session.execute(
            sa.select(Department).where(Department.is_triage.is_(True))
        ).scalar_one_or_none()
    return department


def submit(
    session,
    *,
    raw_text: str,
    principal: Principal = SYSTEM,
    source: IntakeSource = IntakeSource.WEB,
    source_ref: str | None = None,
    student_email: str | None = None,
    student_ref: str | None = None,
    models: dict | None = None,
) -> IntakeResult:
    """Classifies, routes, persists and acknowledges one inquiry.

    Raises DuplicateIntakeError when this exact source message has already
    been ingested. The caller turns that into a skip; it is not an error
    condition, just an idempotent no-op.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValidationError("An inquiry cannot be empty")

    settings = get_settings()
    models = models or registry.get()

    # Idempotency guard for repeated delivery of the same email. The
    # UNIQUE (source, source_ref) constraint is the real enforcement --
    # this check just turns the common case into a clean skip rather than
    # an IntegrityError.
    if source_ref:
        existing = session.execute(
            sa.select(Inquiry.id).where(
                Inquiry.source == source, Inquiry.source_ref == source_ref
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise DuplicateIntakeError(
                f"{source.value} message {source_ref} has already been ingested",
                existing_id=existing,
            )

    active_policy = policy.active_policy(session)
    model_version = policy.active_model_version(session, "category")

    from uniflow.ml.inference import classify_inquiry, embed  # deferred: imports torch

    result = classify_inquiry(
        raw_text,
        models,
        thresholds=active_policy.as_thresholds(),
        calibration=active_policy.as_calibration(),
    )
    decision = RoutingDecision(result["routing_decision"])
    department = _department_for(session, result["category_label"], decision)
    urgency = Urgency(result["urgency_label"])

    now = datetime.now(timezone.utc)
    normalized = result["cleaned_text"]

    inquiry = Inquiry(
        reference="",  # replaced after flush, once the id exists
        raw_text=raw_text,
        normalized_text=normalized,
        text_hash=text_hash(normalized),
        source=source,
        source_ref=source_ref,
        student_email=student_email,
        student_ref=student_ref,
        submitted_by_id=principal.user_id or None,
        department_id=department.id if department else None,
        urgency=urgency,
        status=QueueStatus.NEW,
        routing=decision,
        created_at=now,
        updated_at=now,
    )
    inquiry.sla_due_at = sla.due_at(
        session, urgency, inquiry.department_id, created_at=now
    )
    session.add(inquiry)
    session.flush()
    inquiry.reference = _next_reference(session, inquiry.id, now)

    prediction = Prediction(
        inquiry_id=inquiry.id,
        model_version_id=model_version.id,
        policy_id=active_policy.id,
        category_label=result["category_label"],
        category_confidence=result["category_confidence"],
        category_energy=result["category_energy"],
        category_probs=result["category_probs"],
        urgency_label=result["urgency_label"],
        urgency_confidence=result["urgency_confidence"],
        urgency_energy=result["urgency_energy"],
        urgency_probs=result["urgency_probs"],
        routing=decision,
        abstained=decision.abstained,
        latency_ms=result.get("latency_ms"),
        created_at=now,
    )
    session.add(prediction)
    session.flush()
    inquiry.current_prediction_id = prediction.id

    # ---- duplicate detection ------------------------------------------
    duplicate = None
    exact = dedupe.find_exact_duplicate(
        session,
        inquiry.text_hash,
        lookback_days=active_policy.duplicate_lookback_days,
        exclude_id=inquiry.id,
    )
    if exact is not None:
        duplicate = dedupe.DuplicateMatch(
            inquiry_id=exact.id,
            reference=exact.reference,
            similarity=1.0,
            department_id=exact.department_id,
        )

    vector = embed([raw_text], models)[0].tolist()
    session.add(
        InquiryEmbedding(
            inquiry_id=inquiry.id,
            model_version_id=model_version.id,
            dim=len(vector),
            vector=dedupe.pack(vector),
            created_at=now,
        )
    )

    if duplicate is None:
        duplicate = dedupe.find_duplicate(
            session,
            vector,
            model_version_id=model_version.id,
            threshold=float(active_policy.duplicate_threshold),
            lookback_days=active_policy.duplicate_lookback_days,
            exclude_id=inquiry.id,
        )

    if duplicate is not None:
        inquiry.duplicate_of_id = duplicate.inquiry_id
        inquiry.duplicate_score = round(duplicate.similarity, 5)

    # ---- audit ----------------------------------------------------------
    session.add(
        InquiryEvent(
            inquiry_id=inquiry.id,
            actor_id=principal.user_id or None,
            event_type=EventType.CREATED,
            to_value={
                "source": source.value,
                "routing": decision.value,
                "department": department.code if department else None,
                "urgency": urgency.value,
                "category_confidence": round(result["category_confidence"], 4),
                "policy": active_policy.version,
                "model_version": f"{model_version.name}:{model_version.version}",
                "duplicate_of": duplicate.reference if duplicate else None,
            },
            created_at=now,
        )
    )

    # ---- acknowledgement, via the outbox --------------------------------
    # Written in this same transaction rather than sent inline, so a slow
    # or unavailable SMTP server cannot fail the intake or lose the ack.
    if student_email and settings.send_ack:
        session.add(
            OutboundMessage(
                inquiry_id=inquiry.id,
                to_address=student_email,
                subject=f"[{inquiry.reference}] We have received your inquiry",
                body=_acknowledgement_body(inquiry, result),
                scheduled_for=now,
                created_at=now,
            )
        )

    session.flush()
    return IntakeResult(
        inquiry=inquiry,
        prediction=prediction,
        duplicate=duplicate,
        routing_reason=result["routing_reason"],
    )


def _acknowledgement_body(inquiry: Inquiry, result: dict) -> str:
    """Deliberately says nothing about the predicted department.

    A student should not be told "this went to Examinations" when the
    system may have abstained or been wrong; the acknowledgement confirms
    receipt and the response target, which are both things the system can
    actually commit to.
    """
    due = inquiry.sla_due_at.strftime("%d %b %Y, %H:%M UTC") if inquiry.sla_due_at else "shortly"
    return (
        f"Thank you for contacting the university.\n\n"
        f"Your inquiry has been logged with reference {inquiry.reference}. "
        f"We aim to respond by {due}.\n\n"
        f"Please quote {inquiry.reference} in any follow-up.\n"
    )
