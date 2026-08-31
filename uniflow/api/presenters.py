"""
presenters.py
-------------
ORM objects to response schemas.

SLA state is derived, not stored -- it depends on the current time -- so
it is attached here rather than living on the row. Keeping that in one
place is what stops the four-way classification being reimplemented
slightly differently in each endpoint, which is how the old code ended up
computing it in sla.py, in app.py's card renderer and again in the
analytics page.
"""

from __future__ import annotations

from datetime import datetime

from uniflow.api.schemas import (
    DuplicateOut,
    ExplanationOut,
    InquiryDetail,
    InquiryOut,
    PredictionOut,
)
from uniflow.core.enums import SlaState
from uniflow.db.models import Explanation, Inquiry, Prediction
from uniflow.domain import sla


def prediction_out(prediction: Prediction | None) -> PredictionOut | None:
    """Built field by field rather than via model_validate().

    PredictionOut.model_version is a string ("2026.03.1"); Prediction's own
    attribute of the same name is the related ModelVersion row. Handing the
    ORM object straight to from_attributes validation trips on that name
    collision, so the three provenance fields are read off the relationship
    explicitly instead.
    """
    if prediction is None:
        return None
    return PredictionOut(
        id=prediction.id,
        category_label=prediction.category_label,
        category_confidence=float(prediction.category_confidence),
        category_energy=(
            float(prediction.category_energy)
            if prediction.category_energy is not None else None
        ),
        category_probs=prediction.category_probs,
        urgency_label=prediction.urgency_label,
        urgency_confidence=float(prediction.urgency_confidence),
        urgency_energy=(
            float(prediction.urgency_energy)
            if prediction.urgency_energy is not None else None
        ),
        urgency_probs=prediction.urgency_probs,
        routing=prediction.routing,
        abstained=prediction.abstained,
        latency_ms=prediction.latency_ms,
        created_at=prediction.created_at,
        model_name=prediction.model_version.name if prediction.model_version else None,
        model_version=prediction.model_version.version if prediction.model_version else None,
        policy_version=prediction.policy.version if prediction.policy else None,
    )


def inquiry_out(inquiry: Inquiry, now: datetime | None = None,
                sla_state: SlaState | None = None) -> InquiryOut:
    now = now or sla.utcnow()
    data = InquiryOut.model_validate(inquiry)
    data.sla_state = sla_state or sla.sla_state(inquiry, now)
    data.sla_remaining = sla.time_remaining(inquiry, now)
    return data


def inquiry_detail(inquiry: Inquiry, now: datetime | None = None,
                   routing_reason: str | None = None) -> InquiryDetail:
    now = now or sla.utcnow()
    data = InquiryDetail.model_validate(inquiry)
    data.sla_state = sla.sla_state(inquiry, now)
    data.sla_remaining = sla.time_remaining(inquiry, now)
    data.prediction = prediction_out(inquiry.current_prediction)
    data.routing_reason = routing_reason
    return data


def duplicate_out(match) -> DuplicateOut | None:
    if match is None:
        return None
    return DuplicateOut(
        inquiry_id=match.inquiry_id,
        reference=match.reference,
        similarity=round(match.similarity, 4),
    )


def explanation_out(explanation: Explanation) -> ExplanationOut:
    return ExplanationOut.model_validate(explanation)
