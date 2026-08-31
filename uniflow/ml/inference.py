"""
inference.py
------------
Model loading and inference, including the three research components
layered on top of plain classification:

  1. Temperature scaling      -- calibrated probabilities, so that a
                                 reported 0.80 confidence really does
                                 mean roughly 80% correct.
  2. Selective prediction     -- the system abstains and sends an
                                 inquiry to Triage when calibrated
                                 confidence falls below tau.
  3. Energy-based OOD scoring -- detects inquiries that belong to none
                                 of the seven departments, instead of
                                 forcing them into the closest one.

Moved from app/model_utils.py. Two changes:

  * the models directory comes from settings rather than a module-level
    constant, so tests and alternative deployments can point elsewhere;
  * thresholds and calibration are passed in explicitly. They used to
    default to reading models/thresholds.json and calibration.json off
    disk on every call, which is how a routing decision ended up with no
    record of the policy that produced it.

The returned dictionary shape is unchanged, so evaluation/ keeps working
against it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from uniflow.core.enums import TRIAGE_NAME, RoutingDecision
from uniflow.core.settings import get_settings
from uniflow.ml.multitask import MultiTaskClassifier, mean_pool
from uniflow.ml.preprocessing import clean_text

MAX_LENGTH = 192


def models_dir() -> Path:
    return get_settings().models_dir


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_single_model(task_name: str, root: Path):
    """Loads one fine-tuned model + tokenizer + its ordered label list."""
    model_path = root / f"distilbert_{task_name}"

    if not model_path.exists() or not any(model_path.iterdir()):
        raise FileNotFoundError(
            f"\n\nModel folder missing or empty: {model_path}\n"
            f"Copy your trained 'distilbert_{task_name}' folder into:\n"
            f"  {root}\n"
            f"so that it directly contains model.safetensors, config.json, "
            f"tokenizer.json, and label_classes.json.\n"
        )

    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model.eval()

    label_file = model_path / "label_classes.json"
    mapping_file = model_path / "label_mapping.json"
    if label_file.exists():
        with open(label_file, "r", encoding="utf-8") as f:
            label_classes = json.load(f)
    elif mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        id2label = mapping.get("id2label", {})
        label_classes = [id2label[str(i)] for i in range(model.config.num_labels)]
    else:
        id2label = model.config.id2label
        label_classes = [id2label[i] for i in range(model.config.num_labels)]

    if len(label_classes) != model.config.num_labels:
        raise ValueError(
            f"Label count mismatch in {model_path}: model has "
            f"{model.config.num_labels} outputs but {len(label_classes)} labels were found."
        )

    return model, tokenizer, label_classes


def load_all_models(prefer_multitask: bool | None = None, root: Path | None = None) -> dict:
    """Loads the inference backend.

    Prefers models/multitask/ when it exists (one encoder, two heads),
    otherwise falls back to the two independent models. The returned
    dictionary has the same shape either way, so nothing downstream --
    the services, the evaluation scripts, the explainer -- needs to know
    which backend is active.
    """
    settings = get_settings()
    root = Path(root) if root is not None else settings.models_dir
    if prefer_multitask is None:
        prefer_multitask = settings.prefer_multitask

    multitask_dir = root / "multitask"
    if prefer_multitask and (multitask_dir / "multitask_config.json").exists():
        model, tokenizer, cat_labels, urg_labels = MultiTaskClassifier.load(multitask_dir)
        return {
            "backend": "multitask",
            "artifact_uri": str(multitask_dir),
            "shared": (model, tokenizer),
            "category_labels": cat_labels,
            "urgency_labels": urg_labels,
            "category": (model, tokenizer, cat_labels),
            "urgency": (model, tokenizer, urg_labels),
        }

    category_model, category_tokenizer, category_labels = _load_single_model("category", root)
    urgency_model, urgency_tokenizer, urgency_labels = _load_single_model("urgency", root)
    return {
        "backend": "dual",
        "artifact_uri": str(root),
        "category": (category_model, category_tokenizer, category_labels),
        "urgency": (urgency_model, urgency_tokenizer, urgency_labels),
        "category_labels": category_labels,
        "urgency_labels": urgency_labels,
    }


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def get_logits(texts, models: dict, task: str, batch_size: int = 16) -> torch.Tensor:
    """Raw (uncalibrated) logits for a list of texts.

    Everything else -- calibration fitting, evaluation, OOD scoring --
    is computed from these, so the model is only ever run once per text
    per experiment.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [clean_text(t) for t in texts]

    backend = models.get("backend", "dual")
    if backend == "multitask":
        model, tokenizer = models["shared"]
    else:
        model, tokenizer, _ = models[task]

    collected = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
        if backend == "multitask":
            logits = outputs[f"{task}_logits"]
        else:
            logits = outputs.logits
        collected.append(logits)

    return torch.cat(collected, dim=0)


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Free-energy OOD score, E(x) = -T * logsumexp(logits / T).

    Liu et al. (2020), "Energy-based Out-of-distribution Detection".
    In-distribution inputs produce at least one confidently large logit
    and therefore a *low* (very negative) energy; inputs unlike anything
    in training produce uniformly small logits and higher energy. Unlike
    max-softmax it uses the magnitude of the logits, not just their
    relative sizes, which is why it separates out-of-scope messages that
    still happen to have a peaked softmax.
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def calibrated_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax after dividing logits by the fitted temperature.

    Temperature scaling (Guo et al., 2017) does not change which class
    is predicted -- accuracy is untouched -- it only rescales how
    confident the model claims to be. That property is what makes it a
    safe post-hoc fix rather than a retraining requirement.
    """
    return F.softmax(logits / max(temperature, 1e-6), dim=-1)


def embed(texts, models: dict, batch_size: int = 16) -> torch.Tensor:
    """L2-normalised sentence embeddings from the category encoder.

    Reuses the already-loaded fine-tuned encoder rather than pulling in a
    separate sentence-embedding model. Because the encoder was fine-tuned
    on university inquiries, its representation space is domain-specific,
    which suits near-duplicate detection on this corpus and avoids a
    second few-hundred-megabyte dependency.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [clean_text(t) for t in texts]

    backend = models.get("backend", "dual")
    if backend == "multitask":
        model, tokenizer = models["shared"]
    else:
        model, tokenizer, _ = models["category"]

    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            if backend == "multitask":
                hidden = model.encoder(**inputs).last_hidden_state
            else:
                # Reach past the classification head to the base encoder.
                base = getattr(model, model.base_model_prefix)
                hidden = base(**inputs).last_hidden_state
        pooled = mean_pool(hidden, inputs["attention_mask"])
        vectors.append(F.normalize(pooled, p=2, dim=-1))

    return torch.cat(vectors, dim=0)


# ---------------------------------------------------------------------------
# The routing decision
# ---------------------------------------------------------------------------

def classify_inquiry(raw_text: str, models: dict, thresholds: dict,
                     calibration: dict) -> dict:
    """Classifies one inquiry and decides whether to auto-route it.

    Returns the predicted department and urgency, their calibrated
    confidences, the OOD energy scores, and a routing decision that is
    one of:

      auto          -- confident and in-distribution; goes to the
                       predicted department queue.
      triage        -- calibrated confidence below tau; held for a human
                       to place. This is the abstention that the
                       risk-coverage analysis justifies.
      out_of_scope  -- energy above the OOD threshold; the inquiry
                       probably belongs to no department at all.

    thresholds and calibration are required arguments. They come from the
    active routing_policies row, and the caller records that row's id
    against the prediction it produces.
    """
    started = time.perf_counter()

    cat_labels = models["category_labels"]
    urg_labels = models["urgency_labels"]

    cat_logits = get_logits([raw_text], models, "category")[0]
    urg_logits = get_logits([raw_text], models, "urgency")[0]

    cat_temp = calibration.get("category", {}).get("temperature", 1.0)
    urg_temp = calibration.get("urgency", {}).get("temperature", 1.0)

    cat_probs = calibrated_probs(cat_logits.unsqueeze(0), cat_temp)[0]
    urg_probs = calibrated_probs(urg_logits.unsqueeze(0), urg_temp)[0]

    cat_id = int(torch.argmax(cat_probs))
    urg_id = int(torch.argmax(urg_probs))
    category_label = cat_labels[cat_id]
    urgency_label = urg_labels[urg_id]
    category_conf = float(cat_probs[cat_id])
    urgency_conf = float(urg_probs[urg_id])

    cat_energy = float(energy_score(cat_logits.unsqueeze(0))[0])
    urg_energy = float(energy_score(urg_logits.unsqueeze(0))[0])

    tau = thresholds.get("category", {}).get("tau", 0.0)
    energy_threshold = thresholds.get("category", {}).get("energy_threshold")

    if energy_threshold is not None and cat_energy > energy_threshold:
        decision = RoutingDecision.OUT_OF_SCOPE
        queue = TRIAGE_NAME
        reason = (
            f"Energy score {cat_energy:.2f} exceeds the out-of-scope threshold "
            f"{energy_threshold:.2f}; this inquiry does not resemble the training "
            f"distribution for any department."
        )
    elif category_conf < tau:
        decision = RoutingDecision.TRIAGE
        queue = TRIAGE_NAME
        reason = (
            f"Calibrated confidence {category_conf:.0%} is below the auto-routing "
            f"threshold of {tau:.0%}, so the system abstained rather than risk a "
            f"misroute."
        )
    else:
        decision = RoutingDecision.AUTO
        queue = category_label
        reason = (
            f"Calibrated confidence {category_conf:.0%} meets the auto-routing "
            f"threshold of {tau:.0%}."
        )

    return {
        "cleaned_text": clean_text(raw_text),
        "category_label": category_label,
        "category_confidence": category_conf,
        "category_probs": {l: float(p) for l, p in zip(cat_labels, cat_probs.tolist())},
        "category_energy": cat_energy,
        "urgency_label": urgency_label,
        "urgency_confidence": urgency_conf,
        "urgency_probs": {l: float(p) for l, p in zip(urg_labels, urg_probs.tolist())},
        "urgency_energy": urg_energy,
        "routing_decision": decision,
        "routed_queue": queue,
        "routing_reason": reason,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
