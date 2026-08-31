"""
session.py
----------
Engine and session management.

The old db_utils opened and closed a fresh sqlite3 connection inside every
single function -- roughly fifteen connect/close cycles per page render --
with no WAL mode, no busy timeout and no transaction spanning more than one
statement. Two processes wrote the same file that way, which is why
"database is locked" was inevitable.

Here there is one engine, one connection pool, and a session scope that is
also a transaction boundary: a unit of work either commits whole or rolls
back whole.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from uniflow.core.settings import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _configure_sqlite(engine: Engine) -> None:
    """SQLite needs three PRAGMAs before it behaves under concurrency.

    The application itself only ever runs against PostgreSQL -- this
    exists solely for tests/test_uniflow.py's db_session fixture, which
    builds its own throwaway SQLite engine directly rather than going
    through get_engine() below, so that the test suite stays fast and
    needs no database server running.
    """

    @sa.event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        # Readers no longer block the writer. Without this a single
        # dashboard render could lock out the email poller entirely.
        cursor.execute("PRAGMA journal_mode=WAL")
        # Wait rather than fail immediately when the write lock is held.
        cursor.execute("PRAGMA busy_timeout=5000")
        # SQLite ignores foreign keys unless asked, per connection.
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = sa.create_engine(
            settings.database_url,
            echo=settings.sql_echo,
            future=True,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return _engine


def create_all() -> None:
    """Creates every table that does not already exist.

    There is no migration tool here: this project has no production data
    that a schema change would need to preserve, since scripts.seed.run()
    fully repopulates a fresh database from the model's own label files
    and a fixed set of demo accounts. Changing a model's columns just
    means deleting the database file and re-running this and the seed
    script -- so create_all(), not Alembic, is the whole story.
    """
    from uniflow.db.base import Base
    import uniflow.db.models  # noqa: F401 - registers every table on Base.metadata

    Base.metadata.create_all(get_engine())


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional unit of work.

    Used by jobs and scripts. The API uses the FastAPI dependency in
    uniflow.api.deps, which has the same commit/rollback semantics.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drops the cached engine, e.g. after changing UNIFLOW_DATABASE_URL
    at runtime. Not used by the test suite, which never touches this
    module's cached engine at all."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
