"""
logits_cache.py
---------------
Shared, disk-cached logit computation for the evaluation stages.

Two problems this solves.

Memory: the two transformers together need well over a gigabyte of
resident memory once PyTorch's allocator overhead is counted, and a
typical student laptop running a browser and Streamlit does not have it
free. Loading only the single model a stage actually needs, and
releasing it immediately afterwards, roughly halves peak usage. On a
machine short of RAM the alternative is not slow -- it is a hard crash,
because a failed allocation inside PyTorch surfaces on Windows as an
access violation rather than a clean MemoryError.

Duplicated work: calibration, selective routing and OOD detection are
all functions of the same logits that evaluate.py already computed.
Caching them to .npy means the models are usually not loaded at all
after the first stage, so the whole downstream pipeline runs in seconds
and without touching a transformer.

The cache is keyed by (task, split) and validated by row count, so it
invalidates itself if the split changes size. Delete
evaluation/results/logits_*.npy after retraining a model -- the cache
cannot detect new weights on its own.
"""

import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import RESULTS_DIR  # noqa: E402


def cache_path(task: str, split_name: str) -> Path:
    return RESULTS_DIR / f"logits_{task}_{split_name}.npy"


def load_task_backend(task: str):
    """Loads only the model for one task.

    Returns a dictionary shaped like load_all_models()'s output but
    containing a single task, which is all get_logits() and embed()
    require. Imported lazily so that a stage running entirely from cache
    never pays the cost of importing torch and transformers.
    """
    from policy_files import MODELS_DIR  # noqa: PLC0415
    from uniflow.ml.inference import _load_single_model, load_all_models  # noqa: PLC0415

    if (MODELS_DIR / "multitask" / "multitask_config.json").exists():
        # One shared encoder serves both heads; nothing to save by splitting.
        return load_all_models()

    model, tokenizer, labels = _load_single_model(task, MODELS_DIR)
    return {
        "backend": "dual",
        task: (model, tokenizer, labels),
        f"{task}_labels": labels,
    }


def cached_logits(texts, task: str, split_name: str, models=None, cache: bool = True,
                  verbose: bool = True):
    """Logits for one split, from disk when possible.

    If `models` is supplied it is used; otherwise a single-task backend
    is loaded on demand and freed before returning.
    """
    texts = list(texts)
    path = cache_path(task, split_name)

    if cache and path.exists():
        stored = np.load(path)
        if len(stored) == len(texts):
            if verbose:
                print(f"  reusing cached logits: {path.name}")
            return stored
        if verbose:
            print(f"  cache {path.name} has {len(stored)} rows but the split has "
                  f"{len(texts)}; recomputing")

    from uniflow.ml.inference import get_logits  # noqa: PLC0415

    owned = models is None
    if owned:
        if verbose:
            print(f"  loading {task} model to compute {split_name} logits…")
        models = load_task_backend(task)

    try:
        if verbose:
            print(f"  running {task} model over {len(texts)} {split_name} items…")
        logits = get_logits(texts, models, task).numpy()
    finally:
        if owned:
            # Drop the reference and collect before the next stage loads
            # its own model, so the two never coexist.
            del models
            gc.collect()

    if cache:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, logits)
    return logits


def labels_for(task: str):
    """The model's ordered label list, read without loading the weights.

    Only the config and label files are touched, so a stage that already
    has cached logits can still map class indices to names for a few
    kilobytes rather than several hundred megabytes.
    """
    import json  # noqa: PLC0415

    from policy_files import MODELS_DIR  # noqa: PLC0415

    multitask_config = MODELS_DIR / "multitask" / "multitask_config.json"
    if multitask_config.exists():
        with open(multitask_config, "r", encoding="utf-8") as f:
            return json.load(f)[f"{task}_labels"]

    model_dir = MODELS_DIR / f"distilbert_{task}"
    label_file = model_dir / "label_classes.json"
    if label_file.exists():
        with open(label_file, "r", encoding="utf-8") as f:
            return json.load(f)

    mapping_file = model_dir / "label_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)["id2label"]
        return [mapping[str(i)] for i in range(len(mapping))]

    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        id2label = json.load(f)["id2label"]
    return [id2label[str(i)] for i in range(len(id2label))]
