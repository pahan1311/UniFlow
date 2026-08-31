"""
outbox.py
---------
Delivers queued outbound messages, with retries.

The acknowledgement used to be sent inline during intake with a bare
try/except that printed a warning: a slow SMTP server made the student
wait, and a failure lost the acknowledgement with no record that one had
been owed.

Here the message was written in the same transaction as the inquiry, and
this job delivers it. Failures back off exponentially and give up after
MAX_ATTEMPTS with the error recorded on the row, so an undelivered
acknowledgement is visible rather than silent.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import sqlalchemy as sa

from uniflow.core.enums import MessageStatus
from uniflow.core.settings import get_settings
from uniflow.db.models import OutboundMessage
from uniflow.db.models.messaging import MAX_ATTEMPTS
from uniflow.db.session import session_scope
from uniflow.jobs.locks import job_lock

logger = logging.getLogger(__name__)

BATCH_SIZE = 20


def _send(message: OutboundMessage) -> None:
    settings = get_settings()
    host = settings.smtp_host or settings.imap_host
    user = settings.smtp_user or settings.imap_user
    password = settings.smtp_password or settings.imap_password

    payload = EmailMessage()
    payload["Subject"] = message.subject or "University inquiry"
    payload["From"] = user
    payload["To"] = message.to_address
    payload.set_content(
        (message.body or "") + "\n\n-- UniFlow automated inquiry system\n"
    )

    context = ssl.create_default_context()
    with smtplib.SMTP(host, settings.smtp_port, timeout=30) as server:
        server.starttls(context=context)
        server.login(user, password)
        server.send_message(payload)


def run() -> int:
    """Sends what is due. Returns the number delivered."""
    settings = get_settings()
    if not settings.smtp_enabled:
        return 0

    with job_lock("outbox_drain") as acquired:
        if not acquired:
            return 0

        now = datetime.now(timezone.utc)
        with session_scope() as session:
            due = list(
                session.execute(
                    sa.select(OutboundMessage)
                    .where(
                        OutboundMessage.status == MessageStatus.PENDING,
                        OutboundMessage.scheduled_for <= now,
                    )
                    .order_by(OutboundMessage.scheduled_for)
                    .limit(BATCH_SIZE)
                ).scalars()
            )

            sent = 0
            for message in due:
                message.attempts += 1
                try:
                    _send(message)
                except (smtplib.SMTPException, OSError, ValueError) as exc:
                    message.last_error = str(exc)[:2000]
                    if message.attempts >= MAX_ATTEMPTS:
                        message.status = MessageStatus.FAILED
                        logger.error(
                            "giving up on message %s to %s after %d attempts: %s",
                            message.id, message.to_address, message.attempts, exc,
                        )
                    else:
                        # 1, 2, 4, 8 ... minutes.
                        backoff = 2 ** (message.attempts - 1)
                        message.scheduled_for = now + timedelta(minutes=backoff)
                        logger.warning(
                            "message %s failed (attempt %d), retrying in %dm: %s",
                            message.id, message.attempts, backoff, exc,
                        )
                    continue

                message.status = MessageStatus.SENT
                message.sent_at = datetime.now(timezone.utc)
                message.last_error = None
                sent += 1

    if sent:
        logger.info("outbox: delivered %d message(s)", sent)
    return sent
