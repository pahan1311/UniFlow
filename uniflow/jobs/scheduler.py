"""
scheduler.py
------------
Wires the jobs to intervals inside the API process.

APScheduler rather than Celery: there is no broker to run, no worker to
deploy, and the jobs are all short database operations. The database lock
in locks.py is what makes this safe with more than one API worker, and
the outbox table is where a real queue would slot in if the workload ever
justified one.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from uniflow.core.settings import get_settings
from uniflow.jobs import email_intake, outbox, retention, sla_sweep

logger = logging.getLogger(__name__)


def start() -> BackgroundScheduler:
    settings = get_settings()

    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        sla_sweep.run,
        "interval",
        seconds=settings.sla_sweep_seconds,
        id="sla_sweep",
        # Skip rather than pile up if a sweep overruns its interval.
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        outbox.run,
        "interval",
        seconds=settings.outbox_drain_seconds,
        id="outbox_drain",
        max_instances=1,
        coalesce=True,
    )
    if settings.imap_enabled:
        scheduler.add_job(
            email_intake.run,
            "interval",
            seconds=settings.email_poll_seconds,
            id="email_intake",
            max_instances=1,
            coalesce=True,
        )
    if settings.retention_days:
        scheduler.add_job(retention.run, "cron", hour=3, minute=0, id="retention")

    scheduler.start()
    logger.info(
        "scheduler started: %s", ", ".join(job.id for job in scheduler.get_jobs())
    )
    return scheduler
