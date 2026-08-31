"""
retention.py
------------
Applies a retention window to student personal data.

Inquiry text and the sender's address are personal data. The old system
kept both indefinitely in a local SQLite file, with a one-click "download
all inquiries as CSV" button that exported the lot, and no retention
policy anywhere.

This job redacts the identifying fields on resolved inquiries older than
the configured window while keeping the row, its prediction, its review
events and its timeline. That preserves everything the research and the
audit trail need -- confidences, routing decisions, corrections, SLA
outcomes -- and discards what only identifies a person.

Disabled unless UNIFLOW_RETENTION_DAYS is set, because a deployment
should choose its own window deliberately.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import sqlalchemy as sa

from uniflow.core.settings import get_settings
from uniflow.db.models import Inquiry
from uniflow.db.session import session_scope
from uniflow.domain.sla import utcnow
from uniflow.jobs.locks import job_lock

logger = logging.getLogger(__name__)

REDACTED = "[redacted under the data retention policy]"


def run() -> int:
    settings = get_settings()
    if not settings.retention_days:
        return 0

    with job_lock("retention") as acquired:
        if not acquired:
            return 0

        cutoff = utcnow() - timedelta(days=settings.retention_days)
        with session_scope() as session:
            result = session.execute(
                sa.update(Inquiry)
                .where(
                    Inquiry.resolved_at.isnot(None),
                    Inquiry.resolved_at < cutoff,
                    Inquiry.student_email.isnot(None),
                )
                .values(
                    student_email=None,
                    student_ref=None,
                    raw_text=REDACTED,
                    normalized_text=REDACTED,
                )
            )
            count = result.rowcount

    if count:
        logger.info("redacted personal data on %d inquiries older than %d days",
                    count, settings.retention_days)
    return count
