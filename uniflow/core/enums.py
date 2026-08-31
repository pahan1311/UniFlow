"""
enums.py
--------
The controlled vocabularies of the domain.

In the old schema every one of these was free text with no CHECK
constraint: `role`, `queue_status`, `routing_decision`, urgency and
category were all bare TEXT columns, and departments existed only as
label strings inside a model's label_classes.json. Anything could be
written into any of them.

These enums are the single definition, shared by the SQLAlchemy columns,
the Pydantic schemas and the domain services.
"""

from enum import StrEnum


class Urgency(StrEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    @property
    def rank(self) -> int:
        """Sort order for queues: most urgent first."""
        return {Urgency.HIGH: 0, Urgency.MEDIUM: 1, Urgency.LOW: 2}[self]


class QueueStatus(StrEnum):
    NEW = "New"
    IN_PROGRESS = "In Progress"
    WAITING_ON_STUDENT = "Waiting on Student"
    RESOLVED = "Resolved"
    CLOSED = "Closed"

    @property
    def is_terminal(self) -> bool:
        return self in {QueueStatus.RESOLVED, QueueStatus.CLOSED}


class RoutingDecision(StrEnum):
    """The three outcomes of the selective-prediction policy."""

    AUTO = "auto"           # confident enough to route to a department
    TRIAGE = "triage"       # abstained: confidence below tau
    OUT_OF_SCOPE = "out_of_scope"  # rejected by the energy-based OOD test

    @property
    def abstained(self) -> bool:
        return self is not RoutingDecision.AUTO


class ReviewAction(StrEnum):
    ACCEPT = "accept"
    CORRECT = "correct"


class IntakeSource(StrEnum):
    WEB = "web"
    EMAIL = "email"
    API = "api"
    IMPORT = "import"


class Role(StrEnum):
    ADMIN = "admin"
    OFFICER = "officer"
    VIEWER = "viewer"


class SlaState(StrEnum):
    MET = "met"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    BREACHED = "breached"


class EventType(StrEnum):
    """Entries in the inquiry timeline, which the old schema had no room for."""

    CREATED = "created"
    ROUTED = "routed"
    ASSIGNED = "assigned"
    STATUS_CHANGED = "status_changed"
    ESCALATED = "escalated"
    RESPONDED = "responded"
    RESOLVED = "resolved"
    REOPENED = "reopened"
    MERGED = "merged"
    EXPORTED = "exported"


class MessageStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


# The pseudo-department that holds abstained and out-of-scope inquiries.
# Shared across every officer, because an unrouted inquiry has no owning
# department yet and anyone may claim it.
TRIAGE_CODE = "triage"
TRIAGE_NAME = "Triage"
