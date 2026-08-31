"""
schemas.py
----------
Request and response models.

These are the contract. A future React front end, a student portal or a
university SIS integration reads this file (or the OpenAPI document
generated from it) and needs to know nothing else about UniFlow.

Responses deliberately never include another department's data, and never
include a student's email address unless the caller is looking at a single
inquiry they are entitled to act on.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from uniflow.core.enums import (
    EventType,
    IntakeSource,
    QueueStatus,
    ReviewAction,
    Role,
    RoutingDecision,
    SlaState,
    Urgency,
)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class RefreshRequest(BaseModel):
    refresh_token: str


class DepartmentOut(ORMModel):
    id: int
    code: str
    name: str
    is_triage: bool


class UserOut(ORMModel):
    id: int
    username: str
    display_name: str
    role: Role
    department: DepartmentOut | None = None
    must_change_password: bool = False


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


# ---------------------------------------------------------------------------
# Inquiries
# ---------------------------------------------------------------------------

class InquiryCreate(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    student_email: str | None = Field(default=None, max_length=254)
    student_ref: str | None = Field(default=None, max_length=64)
    source: IntakeSource = IntakeSource.WEB


class PredictionOut(ORMModel):
    id: int
    category_label: str
    category_confidence: float
    category_energy: float | None = None
    category_probs: dict[str, float] | None = None
    urgency_label: str
    urgency_confidence: float
    urgency_energy: float | None = None
    urgency_probs: dict[str, float] | None = None
    routing: RoutingDecision
    abstained: bool
    latency_ms: int | None = None
    created_at: datetime
    # Which model and which thresholds produced this. The old schema
    # recorded neither, so no stored decision could be reproduced.
    model_name: str | None = None
    model_version: str | None = None
    policy_version: str | None = None


class DuplicateOut(BaseModel):
    inquiry_id: int
    reference: str
    similarity: float


class InquiryOut(ORMModel):
    id: int
    reference: str
    raw_text: str
    source: IntakeSource
    status: QueueStatus
    urgency: Urgency | None = None
    routing: RoutingDecision
    department: DepartmentOut | None = None
    assignee: UserOut | None = None
    duplicate_of_id: int | None = None
    duplicate_score: float | None = None
    sla_due_at: datetime | None = None
    sla_state: SlaState | None = None
    sla_remaining: str | None = None
    first_response_at: datetime | None = None
    resolved_at: datetime | None = None
    escalated_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    # Echoed back on reads and required on writes; this is the optimistic
    # lock that stops two officers overwriting each other.
    version: int


class InquiryDetail(InquiryOut):
    student_email: str | None = None
    student_ref: str | None = None
    normalized_text: str
    prediction: PredictionOut | None = None
    routing_reason: str | None = None


class IntakeResponse(BaseModel):
    inquiry: InquiryDetail
    prediction: PredictionOut
    duplicate: DuplicateOut | None = None
    routing_reason: str


class InquiryPage(BaseModel):
    items: list[InquiryOut]
    total: int
    limit: int
    offset: int
    has_more: bool


class StatusUpdate(BaseModel):
    status: QueueStatus | None = None
    assignee_id: int | None = None
    clear_assignee: bool = False
    expected_version: int | None = None


class RouteRequest(BaseModel):
    department_id: int
    note: str | None = Field(default=None, max_length=2000)
    expected_version: int | None = None


class ReviewRequest(BaseModel):
    action: ReviewAction
    corrected_department_id: int | None = None
    corrected_urgency: Urgency | None = None
    note: str | None = Field(default=None, max_length=2000)
    expected_version: int | None = None


class ReviewOut(ORMModel):
    id: int
    inquiry_id: int
    prediction_id: int | None = None
    action: ReviewAction
    corrected_department_id: int | None = None
    corrected_urgency: Urgency | None = None
    note: str | None = None
    created_at: datetime
    reviewer: UserOut | None = None


class EventOut(ORMModel):
    id: int
    event_type: EventType
    from_value: dict | None = None
    to_value: dict | None = None
    created_at: datetime
    actor: UserOut | None = None


class TokenAttribution(BaseModel):
    token: str
    attribution: float


class ExplanationOut(ORMModel):
    prediction_id: int
    task: str
    method: str
    steps: int | None = None
    tokens: list[TokenAttribution]
    keyword_agreement: float | None = None
    created_at: datetime


class ClassifyRequest(BaseModel):
    """Stateless classification: no inquiry is created, nothing is stored.

    Exists so evaluation/benchmark.py and any external caller can exercise
    the exact production routing policy without polluting the queue.
    """

    text: str = Field(min_length=1, max_length=20_000)


class ClassifyResponse(BaseModel):
    category_label: str
    category_confidence: float
    category_energy: float | None = None
    category_probs: dict[str, float]
    urgency_label: str
    urgency_confidence: float
    urgency_probs: dict[str, float]
    routing_decision: RoutingDecision
    routing_reason: str
    latency_ms: int | None = None
    policy_version: str
    model_version: str


# ---------------------------------------------------------------------------
# Admin / registry
# ---------------------------------------------------------------------------

class ModelVersionOut(ORMModel):
    id: int
    name: str
    version: str
    backend: str
    artifact_uri: str
    label_classes: list[str]
    metrics: dict | None = None
    dataset_hash: str | None = None
    split_hash: str | None = None
    trained_at: datetime | None = None
    is_active: bool
    promoted_at: datetime | None = None


class RoutingPolicyOut(ORMModel):
    id: int
    version: str
    category_tau: float
    category_temperature: float
    category_energy_threshold: float | None = None
    urgency_tau: float
    urgency_temperature: float
    duplicate_threshold: float
    duplicate_lookback_days: int
    derived_from: dict | None = None
    is_active: bool
    activated_at: datetime | None = None
    created_at: datetime


class RoutingPolicyCreate(BaseModel):
    version: str = Field(min_length=1, max_length=80)
    category_tau: float = Field(ge=0, le=1)
    category_temperature: float = Field(gt=0)
    category_energy_threshold: float | None = None
    urgency_tau: float = Field(ge=0, le=1)
    urgency_temperature: float = Field(gt=0)
    duplicate_threshold: float = Field(default=0.92, ge=0, le=1)
    duplicate_lookback_days: int = Field(default=30, gt=0)
    derived_from: dict | None = None
    activate: bool = False


class HealthOut(BaseModel):
    status: str
    database: bool
    models: dict
    version: str
