"""
locks.py
--------
A cross-process lock so a job runs once per interval, not once per worker.

pg_try_advisory_lock costs nothing to acquire, needs no table, and is
released automatically when the connection ends -- including if the
process dies, so a crashed holder can never leave the lock stuck.
"""

from __future__ import annotations

import logging
import zlib
from contextlib import contextmanager

import sqlalchemy as sa

from uniflow.db.session import get_engine

logger = logging.getLogger(__name__)


def _key(name: str) -> int:
    # PostgreSQL advisory locks are keyed by bigint, so the job name is
    # hashed into one. Signed 32-bit keeps it inside the two-int form too.
    return zlib.crc32(name.encode("utf-8")) - 2**31


@contextmanager
def job_lock(name: str):
    """Yields True if this process took the lock, False if someone else holds it."""
    engine = get_engine()
    with engine.connect() as connection:
        acquired = connection.execute(
            sa.text("SELECT pg_try_advisory_lock(:key)"), {"key": _key(name)}
        ).scalar()
        try:
            yield bool(acquired)
        finally:
            if acquired:
                connection.execute(
                    sa.text("SELECT pg_advisory_unlock(:key)"), {"key": _key(name)}
                )
                connection.commit()
