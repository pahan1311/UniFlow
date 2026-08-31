"""UniFlow — university inquiry routing with selective prediction.

The package is layered deliberately:

    api/      HTTP surface. Routers and Pydantic schemas only; no business
              rules, no SQL, no torch.
    domain/   Every business rule lives here. Services take a session and a
              principal and return plain objects.
    db/       SQLAlchemy models and repositories. The only place SQL is written.
    ml/       Model loading, inference, explainability, embeddings.
    jobs/     Scheduled work (SLA sweep, email intake, outbox drain).
    core/     Settings, security primitives, errors, logging.

Dependencies point inward: api -> domain -> db/ml. Nothing in domain imports
from api, and nothing in db imports from domain.
"""

__version__ = "2.0.0"
