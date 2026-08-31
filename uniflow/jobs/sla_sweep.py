"""
sla_sweep.py
------------
Escalates breached tickets on a schedule.

The README's first stated limitation was that escalation is opportunistic:
"Breached tickets are flagged when a page is loaded, not by a background
scheduler. A single-process Streamlit deployment has nowhere to run one."
The API process has somewhere to run one.

The whole sweep is two statements: one indexed SELECT over the partial
index on unresolved rows, and one set-based UPDATE. The old version pulled
every inquiry into a DataFrame, iterated it row by row calling sla_state()
on each, then issued one UPDATE per breach.
"""

from __future__ import annotations

import logging

from uniflow.db.repositories import inquiries as repo
from uniflow.db.session import session_scope
from uniflow.domain import workflow
from uniflow.jobs.locks import job_lock

logger = logging.getLogger(__name__)


def run() -> int:
    """Escalates every unresolved, breached, not-yet-escalated ticket."""
    with job_lock("sla_sweep") as acquired:
        if not acquired:
            logger.debug("sla_sweep already running elsewhere")
            return 0

        with session_scope() as session:
            ids = repo.breached_ids(session)
            if not ids:
                return 0
            # actor_id stays None: the timeline distinguishes "the system
            # escalated this" from "a person escalated this".
            count = workflow.escalate(session, ids, actor_id=None)

        if count:
            logger.info("escalated %d breached inquiries", count)
        return count
