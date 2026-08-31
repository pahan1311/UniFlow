"""
explanations.py
---------------
Serving Integrated Gradients attributions, computed once.

The old flow: render the "Why this route" tab, run ~32 forward+backward
passes synchronously while the officer waits, cache the result in
st.session_state, and lose it on the next reload. Meanwhile the
`explanation` column on inquiries was never written by any code path.

Here the attribution is keyed on the prediction it explains -- an
explanation of a superseded prediction is not an explanation of the
current one -- and stored, so the second officer to open the same ticket
pays nothing.
"""

from __future__ import annotations

import sqlalchemy as sa

from uniflow.core.errors import NotFoundError
from uniflow.db.models import Explanation, Inquiry, Prediction
from uniflow.domain.principal import Principal, assert_can_read

DEFAULT_STEPS = 32


def get_or_compute(session, principal: Principal, inquiry_id: int, *,
                   task: str = "category", models: dict | None = None,
                   steps: int = DEFAULT_STEPS) -> Explanation:
    inquiry = session.get(Inquiry, inquiry_id)
    if inquiry is None:
        raise NotFoundError(f"Inquiry {inquiry_id} does not exist")
    assert_can_read(session, principal, inquiry)

    prediction: Prediction | None = inquiry.current_prediction
    if prediction is None:
        raise NotFoundError(f"Inquiry {inquiry.reference} has no prediction to explain")

    cached = session.execute(
        sa.select(Explanation).where(
            Explanation.prediction_id == prediction.id,
            Explanation.task == task,
            Explanation.method == "integrated_gradients",
        )
    ).scalar_one_or_none()
    if cached is not None:
        return cached

    from uniflow.ml import registry
    from uniflow.ml.explain import (
        as_token_rows,
        integrated_gradients,
        keyword_agreement,
    )

    models = models or registry.get()
    tokens, scores, _ = integrated_gradients(
        inquiry.raw_text, models, task=task, steps=steps
    )
    agreement = keyword_agreement(inquiry.raw_text, tokens, scores)

    explanation = Explanation(
        prediction_id=prediction.id,
        task=task,
        method="integrated_gradients",
        steps=steps,
        tokens=as_token_rows(tokens, scores),
        keyword_agreement=round(agreement["jaccard"], 3),
    )
    session.add(explanation)
    session.flush()
    return explanation
