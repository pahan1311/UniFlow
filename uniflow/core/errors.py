"""
errors.py
---------
Domain-level exceptions.

Services raise these; the API layer is the only place that knows they map
to HTTP status codes. That keeps the domain callable from a job, a test or
a script without dragging FastAPI along.
"""


class UniFlowError(Exception):
    """Base class for every error the domain raises deliberately."""

    status_code = 400

    def __init__(self, message: str, *, detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class NotFoundError(UniFlowError):
    status_code = 404


class PermissionDeniedError(UniFlowError):
    """Raised when a principal may not see or act on a resource.

    The old system had no equivalent: authorization was a pandas filter
    applied after the whole table had already been read, and write handlers
    took a raw inquiry id with no ownership check at all.
    """

    status_code = 403


class AuthenticationError(UniFlowError):
    status_code = 401


class ConflictError(UniFlowError):
    """Optimistic-locking failure, or a uniqueness constraint violation."""

    status_code = 409


class DuplicateIntakeError(ConflictError):
    """The same source message has already been ingested.

    Carries the existing inquiry id so the caller can report a skip rather
    than an error.
    """

    def __init__(self, message: str, *, existing_id: int | None = None):
        super().__init__(message, detail={"existing_id": existing_id})
        self.existing_id = existing_id


class ValidationError(UniFlowError):
    status_code = 422


class ModelUnavailableError(UniFlowError):
    status_code = 503
