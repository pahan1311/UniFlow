"""
conftest.py
-----------
Satisfies uniflow.core.settings.Settings.database_url, which has no
default now that the application is PostgreSQL-only, so that any test
touching get_settings() -- even for an unrelated field like secret_key
or max_failed_logins -- doesn't fail before it starts.

This value is never actually dialled: every database test builds its own
independent SQLite engine directly (see the db_session fixture in
test_uniflow.py) rather than going through get_engine(), which is the
only thing that would open a real connection to it.
"""

import os

os.environ.setdefault(
    "UNIFLOW_DATABASE_URL", "postgresql+psycopg://unused:unused@localhost/unused"
)
