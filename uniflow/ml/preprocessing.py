"""
preprocessing.py
----------------
Text cleaning logic. This MUST match whatever cleaning you did before
training your models — if training saw cleaned text but the dashboard
feeds raw text, predictions will be less accurate.

Moved verbatim from app/preprocessing.py. Any change here silently
invalidates every stored prediction, which is why Inquiry.normalized_text
records this function's output alongside the raw text.
"""

import re


def clean_text(value: str) -> str:
    """Lowercase-safe cleaning: normalizes quotes and whitespace.

    Note: we do not lowercase here because the uncased model tokenizers
    already normalize casing internally.
    """
    if value is None:
        return ""
    value = str(value)
    value = value.replace("’", "'").replace("‘", "'")  # smart quotes -> normal
    value = re.sub(r"\s+", " ", value)  # collapse whitespace
    return value.strip()
