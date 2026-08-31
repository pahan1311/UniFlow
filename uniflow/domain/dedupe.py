"""
dedupe.py
---------
Near-duplicate detection over recent inquiries.

Students frequently re-send the same question by a second channel, or a
department forwards a message that is already in the queue. Two officers
then work the same problem.

The old implementation pulled every inquiry from the last 30 days and
re-embedded all of them through the transformer on *every single
submission* -- N forward passes per intake, growing without bound. Here
the vector is computed once, at intake, and stored; a submission costs one
forward pass and a similarity scan over stored vectors.

Vectors are stored as raw float32 bytes so the same code runs on SQLite
and PostgreSQL. The scan below is exact and adequate at the scale this
system operates at; on PostgreSQL the column can be migrated to
pgvector's vector(dim) with an HNSW index, and only `find_duplicate`
changes -- callers ask for a nearest neighbour and do not care how it was
found.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import sqlalchemy as sa

from uniflow.db.models import Inquiry, InquiryEmbedding
from uniflow.domain.sla import utcnow

DTYPE = np.float32


def pack(vector) -> bytes:
    """float32 little-endian bytes, L2-normalised."""
    array = np.asarray(vector, dtype=DTYPE).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm > 0:
        array = array / norm
    return array.astype(DTYPE).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=DTYPE)


@dataclass(frozen=True)
class DuplicateMatch:
    inquiry_id: int
    reference: str
    similarity: float
    department_id: int | None


def find_duplicate(session, vector, *, model_version_id: int, threshold: float,
                   lookback_days: int, exclude_id: int | None = None
                   ) -> DuplicateMatch | None:
    """The most similar recent inquiry, if it clears the threshold.

    Candidates are scoped to the same model version: vectors from
    different encoders are not comparable, and silently mixing them would
    produce meaningless cosine scores.
    """
    cutoff = utcnow() - timedelta(days=lookback_days)

    query = (
        sa.select(
            InquiryEmbedding.inquiry_id,
            InquiryEmbedding.vector,
            Inquiry.reference,
            Inquiry.department_id,
        )
        .join(Inquiry, Inquiry.id == InquiryEmbedding.inquiry_id)
        .where(
            InquiryEmbedding.model_version_id == model_version_id,
            Inquiry.created_at >= cutoff,
        )
    )
    if exclude_id is not None:
        query = query.where(InquiryEmbedding.inquiry_id != exclude_id)

    rows = session.execute(query).all()
    if not rows:
        return None

    probe = np.asarray(vector, dtype=DTYPE).reshape(-1)
    norm = float(np.linalg.norm(probe))
    if norm > 0:
        probe = probe / norm

    matrix = np.vstack([unpack(row.vector) for row in rows])
    if matrix.shape[1] != probe.shape[0]:
        # A dimension change means the encoder changed without the model
        # version changing. Refusing to score is better than comparing
        # incomparable vectors.
        return None

    # Both sides are unit vectors, so the dot product is the cosine.
    similarities = matrix @ probe
    best = int(np.argmax(similarities))
    score = float(similarities[best])
    if score < threshold:
        return None

    row = rows[best]
    return DuplicateMatch(
        inquiry_id=row.inquiry_id,
        reference=row.reference,
        similarity=score,
        department_id=row.department_id,
    )


def find_exact_duplicate(session, text_hash: str, *, lookback_days: int,
                         exclude_id: int | None = None) -> Inquiry | None:
    """An identical message, found by hash rather than by transformer.

    A resend is usually byte-identical after normalisation, so this
    catches the common case for the cost of an index lookup -- no model
    involved at all.
    """
    cutoff = utcnow() - timedelta(days=lookback_days)
    query = sa.select(Inquiry).where(
        Inquiry.text_hash == text_hash,
        Inquiry.created_at >= cutoff,
    )
    if exclude_id is not None:
        query = query.where(Inquiry.id != exclude_id)
    return session.execute(query.order_by(Inquiry.id.desc()).limit(1)).scalar_one_or_none()
