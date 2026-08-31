"""Persistence: SQLAlchemy models, session management, repositories.

This is the only layer that writes SQL. Everything above it receives ORM
objects or plain dataclasses.
"""

from uniflow.db.base import Base

__all__ = ["Base"]
