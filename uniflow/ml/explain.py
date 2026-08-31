"""
explain.py
----------
Integrated Gradients attribution over the actual classifier.

This replaces the hard-coded keyword list as the system's explanation
mechanism. keywords.py is retained, but demoted to what it always was --
a human-authored heuristic -- and is now used as a *comparison* baseline:
keyword_agreement() below measures how often the model's own evidence
coincides with the words a human would have highlighted, which is itself
a reportable result rather than a decoration.

Method (Sundararajan et al., 2017): attribution for token i is

    (e_i - e'_i) * (1/m) * sum_k dF/de  evaluated at  e' + (k/m)(e - e')

where e is the token's embedding, e' the corresponding baseline
embedding, F the logit of the predicted class, and m the number of
interpolation steps. Summing over the embedding dimension gives one
signed score per token: positive means the token pushed the model
towards the predicted class.

The baseline is the all-[PAD] sequence with special tokens preserved,
which is the standard "absence of information" reference for text.

Moved from app/explain.py. The maths is unchanged; what changed is where
the result goes. This costs roughly 32 forward+backward passes, and it
used to run synchronously during page rendering with the result cached
only in st.session_state, so it was recomputed after every reload. The
explanations table now stores it once per prediction.

highlight_html() stayed behind in the UI layer, where markup belongs.
"""

import torch

from uniflow.ml.inference import MAX_LENGTH
from uniflow.ml.keywords import find_trigger_keywords
from uniflow.ml.preprocessing import clean_text

DEFAULT_STEPS = 32


def _forward_logits(model, backend: str, task: str, inputs_embeds, attention_mask):
    """Runs the right forward pass for whichever backend is loaded."""
    if backend == "multitask":
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out[f"{task}_logits"]
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits


def integrated_gradients(text: str, models: dict, task: str = "category",
                         target_index: int | None = None, steps: int = DEFAULT_STEPS):
    """Per-token attributions for one inquiry.

    Returns (tokens, scores, predicted_index) where scores are signed and
    normalised to [-1, 1] by the largest absolute attribution, so they
    can be mapped straight onto highlight opacities in the UI.
    """
    backend = models.get("backend", "dual")
    if backend == "multitask":
        model, tokenizer = models["shared"]
    else:
        model, tokenizer, _ = models[task]

    encoded = tokenizer(
        clean_text(text), truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    embedding_layer = model.get_input_embeddings()
    input_embeds = embedding_layer(input_ids).detach()

    # Baseline: every ordinary token replaced by [PAD], special tokens kept
    # in place so the sequence stays structurally valid.
    special_ids = set(tokenizer.all_special_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids = input_ids.clone()
    for position, token_id in enumerate(input_ids[0].tolist()):
        if token_id not in special_ids:
            baseline_ids[0, position] = pad_id
    baseline_embeds = embedding_layer(baseline_ids).detach()

    if target_index is None:
        with torch.no_grad():
            logits = _forward_logits(model, backend, task, input_embeds, attention_mask)
        target_index = int(torch.argmax(logits, dim=-1)[0])

    difference = input_embeds - baseline_embeds
    accumulated = torch.zeros_like(input_embeds)

    # Riemann midpoint approximation of the path integral.
    for step in range(steps):
        alpha = (step + 0.5) / steps
        point = (baseline_embeds + alpha * difference).clone().requires_grad_(True)
        logits = _forward_logits(model, backend, task, point, attention_mask)
        score = logits[0, target_index]
        model.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(score, point)[0]
        accumulated += grad.detach()

    attributions = (difference * accumulated / steps).sum(dim=-1)[0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = attributions.tolist()

    tokens, scores = _merge_wordpieces(tokens, scores, tokenizer)

    largest = max((abs(s) for s in scores), default=0.0)
    if largest > 0:
        scores = [s / largest for s in scores]

    return tokens, scores, target_index


def _merge_wordpieces(tokens, scores, tokenizer):
    """Joins subword pieces back into words and drops special tokens.

    Attribution of a word is the sum of its pieces' attributions, which
    keeps the total attributed mass unchanged while making the highlight
    readable to a department officer.
    """
    specials = set(tokenizer.all_special_tokens)
    merged_tokens, merged_scores = [], []
    for token, score in zip(tokens, scores):
        if token in specials:
            continue
        if token.startswith("##") and merged_tokens:
            merged_tokens[-1] += token[2:]
            merged_scores[-1] += score
        else:
            merged_tokens.append(token)
            merged_scores.append(score)
    return merged_tokens, merged_scores


def top_evidence(tokens, scores, k: int = 6):
    """The k words that most supported the prediction, strongest first."""
    ranked = sorted(
        (pair for pair in zip(tokens, scores) if pair[1] > 0),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:k]


def keyword_agreement(text: str, tokens, scores, k: int = 6) -> dict:
    """Compares model evidence against the human keyword heuristic.

    Reported in the thesis as a sanity check on explanation quality: if
    the model's top-weighted tokens systematically ignore the words a
    human would flag, either the explanation or the model deserves
    scrutiny. Returns the overlap and both sides of the comparison.
    """
    heuristic = find_trigger_keywords(text)
    heuristic_words = {w for phrase in heuristic for w in phrase.lower().split()}
    model_words = {token.lower() for token, _ in top_evidence(tokens, scores, k)}
    overlap = sorted(heuristic_words & model_words)
    union = heuristic_words | model_words
    return {
        "heuristic_keywords": heuristic,
        "model_top_words": sorted(model_words),
        "overlap": overlap,
        "jaccard": len(overlap) / len(union) if union else 0.0,
    }


def as_token_rows(tokens, scores) -> list[dict]:
    """Shape stored in explanations.tokens and returned by the API."""
    return [
        {"token": token, "attribution": round(float(score), 6)}
        for token, score in zip(tokens, scores)
    ]
