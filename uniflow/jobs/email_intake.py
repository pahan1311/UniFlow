"""
email_intake.py
---------------
Automatic inquiry intake over IMAP.

This is the component that makes the claim "automatic" defensible.
Without it, a human still has to read each inquiry and paste it into a
web form, which means the system automates classification but not intake.

The parsing is carried over unchanged from app/ingest_email.py -- quoted
reply chains and signatures are stripped because they would otherwise
dominate the 192-token window and the model would classify the wrong
message. What changed is everything after parsing:

  * it calls triage.submit(), the same service the web form calls,
    instead of holding a second copy of the intake flow that had already
    started to drift;
  * it runs inside the API process on a schedule, so there is no separate
    unsupervised `while True: sleep(60)` process writing to the same
    database file with its own gigabyte of model weights;
  * the acknowledgement goes to the outbox in the same transaction as the
    inquiry, rather than being sent inline with a bare try/except that
    printed a warning and dropped the acknowledgement on failure.
"""

from __future__ import annotations

import email
import imaplib
import logging
import re
from email.header import decode_header, make_header

from uniflow.core.enums import IntakeSource
from uniflow.core.errors import DuplicateIntakeError, ValidationError
from uniflow.core.settings import get_settings
from uniflow.db.session import session_scope
from uniflow.domain import triage
from uniflow.domain.principal import SYSTEM
from uniflow.jobs.locks import job_lock

logger = logging.getLogger(__name__)


def decode(value) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except (UnicodeDecodeError, LookupError, ValueError):
        return str(value)


def extract_body(message) -> str:
    """Plain-text body with quoted history and signatures stripped."""
    body = ""
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(
                part.get("Content-Disposition") or ""
            ):
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
                    break
    else:
        payload = message.get_payload(decode=True)
        if payload:
            charset = message.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")

    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            continue
        if re.match(r"^On .+ wrote:$", stripped):
            break
        if stripped in {"--", "-- "}:
            break
        if re.match(r"^-{3,}\s*(Original Message|Forwarded message)", stripped, re.I):
            break
        lines.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def build_inquiry_text(subject: str, body: str) -> str:
    """Subject and body joined, since the subject often carries the topic."""
    subject = re.sub(r"^\s*(re|fwd?)\s*:\s*", "", subject or "", flags=re.I).strip()
    if subject and body:
        return f"{subject}. {body}"
    return subject or body


# Local-part patterns that mean "this mailbox doesn't take replies", not
# "this is a student". A dedicated inquiries inbox otherwise ingests its
# own account-security notifications as if they were real inquiries --
# which is exactly what happened the first time this poller ran here.
_AUTOMATED_SENDER_PATTERNS = (
    "no-reply", "noreply", "do-not-reply", "donotreply",
    "notification", "notifications", "mailer-daemon", "postmaster",
)


def is_automated_sender(address: str | None) -> bool:
    """True if the sender's local part looks like an automated mailbox."""
    if not address or "@" not in address:
        return False
    local_part = address.split("@", 1)[0].lower()
    return any(pattern in local_part for pattern in _AUTOMATED_SENDER_PATTERNS)


def parse_message(raw_bytes: bytes) -> dict:
    """Everything the intake service needs, extracted from one RFC822 message."""
    message = email.message_from_bytes(raw_bytes)
    message_id = decode(message.get("Message-ID")) or decode(message.get("Date"))
    sender = decode(message.get("From"))
    subject = decode(message.get("Subject"))

    address = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", sender)
    return {
        "message_id": message_id or None,
        "student_email": address.group(0) if address else None,
        "text": build_inquiry_text(subject, extract_body(message)),
    }


def ingest_one(raw_bytes: bytes, dry_run: bool = False) -> dict:
    """Classifies and stores one fetched email. Returns a summary dict."""
    parsed = parse_message(raw_bytes)

    if is_automated_sender(parsed["student_email"]):
        return {
            "status": "skipped_automated_sender",
            "message_id": parsed["message_id"],
            "sender": parsed["student_email"],
        }

    if not parsed["text"].strip():
        return {"status": "skipped_empty", "message_id": parsed["message_id"]}

    if dry_run:
        return {"status": "dry_run", "text": parsed["text"][:120],
                "message_id": parsed["message_id"]}

    try:
        with session_scope() as session:
            result = triage.submit(
                session,
                raw_text=parsed["text"],
                principal=SYSTEM,
                source=IntakeSource.EMAIL,
                source_ref=parsed["message_id"],
                student_email=parsed["student_email"],
            )
            return {
                "status": "ingested",
                "id": result.inquiry.id,
                "reference": result.inquiry.reference,
                "routing": result.inquiry.routing.value,
                "duplicate_of": result.duplicate.reference if result.duplicate else None,
            }
    except DuplicateIntakeError as exc:
        # Not an error: the same message arriving twice is expected, and
        # the UNIQUE (source, source_ref) constraint makes it a no-op.
        return {"status": "skipped_duplicate", "existing_id": exc.existing_id}
    except ValidationError as exc:
        return {"status": "skipped_invalid", "detail": exc.message}


def poll_once(dry_run: bool = False, limit: int = 25) -> list[dict]:
    """One IMAP pass over unseen messages."""
    settings = get_settings()
    if not settings.imap_enabled:
        raise RuntimeError(
            "IMAP is not configured. Set UNIFLOW_IMAP_HOST, UNIFLOW_IMAP_USER and "
            "UNIFLOW_IMAP_PASSWORD in .env. See .env.example."
        )

    results: list[dict] = []
    with imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port) as client:
        client.login(settings.imap_user, settings.imap_password)
        client.select(settings.imap_folder)
        status, data = client.search(None, "UNSEEN")
        if status != "OK":
            return results

        for message_number in data[0].split()[:limit]:
            # BODY.PEEK does not set the \Seen flag, so a dry run leaves
            # the mailbox exactly as it found it.
            fetch_flag = "(BODY.PEEK[])" if dry_run else "(RFC822)"
            status, payload = client.fetch(message_number, fetch_flag)
            if status != "OK" or not payload or not isinstance(payload[0], tuple):
                continue
            results.append(ingest_one(payload[0][1], dry_run))

    return results


def run() -> int:
    """Scheduled entry point. Never raises: a mail outage must not kill the job."""
    settings = get_settings()
    if not settings.imap_enabled:
        return 0

    with job_lock("email_intake") as acquired:
        if not acquired:
            return 0
        try:
            results = poll_once()
        except (imaplib.IMAP4.error, OSError, RuntimeError):
            logger.exception("email poll failed; will retry on the next tick")
            return 0

    ingested = sum(1 for r in results if r["status"] == "ingested")
    skipped_automated = sum(1 for r in results if r["status"] == "skipped_automated_sender")
    if results:
        logger.info(
            "email intake: %d fetched, %d ingested, %d from automated senders skipped",
            len(results), ingested, skipped_automated,
        )
    return ingested
