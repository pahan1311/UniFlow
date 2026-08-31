"""
keywords.py
-----------
Heuristic keyword lists used ONLY for the explainability comparison
(highlighting why an inquiry might look urgent to a human reviewer). This
is NOT used by the model itself — the model makes its own prediction
independently. This is a transparency aid, and the thesis should describe
it exactly that way: heuristic, not model attribution.

Moved verbatim from app/keywords.py.
"""

HIGH_URGENCY_KEYWORDS = [
    "today", "urgent", "urgently", "asap", "immediately",
    "cannot access", "can't access", "cannot log in", "can't log in",
    "blocked", "locked out", "exam tomorrow", "exam today",
    "deadline today", "due tonight", "right now", "no workaround",
]

MEDIUM_URGENCY_KEYWORDS = [
    "deadline", "this week", "soon", "tomorrow", "schedule conflict",
    "within a few days", "before the deadline",
]


def find_trigger_keywords(text: str) -> list[str]:
    """Returns which keywords (if any) were found in the given text.

    Case-insensitive substring match — simple and fully transparent,
    exactly the kind of heuristic the report describes.
    """
    lowered = (text or "").lower()
    matches = []
    for kw in HIGH_URGENCY_KEYWORDS + MEDIUM_URGENCY_KEYWORDS:
        if kw in lowered:
            matches.append(kw)
    return matches
