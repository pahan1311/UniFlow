"""
model_utils.py
--------------
Handles loading the saved category DistilBERT and urgency BERT models
and running predictions. Loading is wrapped with @st.cache_resource in
app.py so the ~250MB model files are only read from disk ONCE per
server run, not on every button click.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import clean_text

# Project root (one level up from this file, i.e. university-inquiry-dashboard/)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def _load_single_model(task_name: str):
    """Loads one fine-tuned model + tokenizer + its label list.

    task_name: "category" or "urgency"
    Expects a folder at models/distilbert_<task_name>/ containing the
    files produced by trainer.save_model() + tokenizer.save_pretrained()
    + a label_classes.json file.
    """
    model_path = MODELS_DIR / f"distilbert_{task_name}"

    if not model_path.exists() or not any(model_path.iterdir()):
        raise FileNotFoundError(
            f"\n\nModel folder missing or empty: {model_path}\n"
            f"Copy your trained 'distilbert_{task_name}' folder "
            f"(from Google Drive / your downloaded zip) into:\n"
            f"  {MODELS_DIR}\n"
            f"so that it directly contains model.safetensors, config.json, "
            f"tokenizer.json, and label_classes.json.\n"
        )

    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model.eval()  # inference mode: disables dropout etc.

    # Support both the original list export and the newer BERT mapping export.
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


def load_all_models():
    """Loads both models at once. Called once via st.cache_resource."""
    category_model, category_tokenizer, category_labels = _load_single_model("category")
    urgency_model, urgency_tokenizer, urgency_labels = _load_single_model("urgency")
    return {
        "category": (category_model, category_tokenizer, category_labels),
        "urgency": (urgency_model, urgency_tokenizer, urgency_labels),
    }


def _predict(text: str, model, tokenizer, label_classes, max_length: int = 192):
    """Runs one forward pass and returns (predicted_label, confidence, all_probs)."""
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():  # no gradient tracking needed for inference -> faster, less memory
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()

    predicted_id = int(torch.argmax(probs))
    confidence = float(probs[predicted_id])
    all_probs = {label: float(p) for label, p in zip(label_classes, probs.tolist())}
    return label_classes[predicted_id], confidence, all_probs


def classify_inquiry(raw_text: str, models: dict):
    """Classify both department and urgency directly from inquiry text."""
    cleaned = clean_text(raw_text)

    cat_model, cat_tokenizer, cat_labels = models["category"]
    urg_model, urg_tokenizer, urg_labels = models["urgency"]

    category_label, category_conf, category_probs = _predict(cleaned, cat_model, cat_tokenizer, cat_labels)
    urgency_label, urgency_conf, urgency_probs = _predict(cleaned, urg_model, urg_tokenizer, urg_labels)

    return {
        "cleaned_text": cleaned,
        "category_label": category_label,
        "category_confidence": category_conf,
        "category_probs": category_probs,
        "urgency_label": urgency_label,
        "urgency_confidence": urgency_conf,
        "urgency_probs": urgency_probs,
    }
