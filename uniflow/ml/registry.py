"""
registry.py
-----------
The process-wide handle on the loaded models.

The weights are loaded exactly once, during API startup, and every
request shares them. Previously @st.cache_resource achieved that per
Streamlit *process* -- but the email poller was a second process holding
its own full copy, so the deployment carried the ~1 GB twice and the
README had to warn about Windows allocation failures.

The registry also reconciles what is on disk with the model_versions
table, so that a prediction can always name the artefact that produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

from uniflow.core.errors import ModelUnavailableError
from uniflow.core.settings import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_models: dict | None = None
_load_error: Exception | None = None


def load(force: bool = False) -> dict:
    """Loads the backend, or returns the already-loaded one."""
    global _models, _load_error
    with _lock:
        if _models is not None and not force:
            return _models
        from uniflow.ml.inference import load_all_models  # deferred: imports torch

        try:
            _models = load_all_models()
            _load_error = None
            logger.info(
                "loaded %s backend with %d departments and %d urgency levels",
                _models["backend"],
                len(_models["category_labels"]),
                len(_models["urgency_labels"]),
            )
        except Exception as exc:  # noqa: BLE001 - reported through /readyz
            _load_error = exc
            logger.exception("model loading failed")
            raise
        return _models


def get() -> dict:
    """The loaded backend, or a 503-shaped error.

    Deliberately does not load on demand: a lazy first request would pay
    several seconds and a gigabyte of allocation while holding a database
    transaction open.
    """
    if _models is None:
        raise ModelUnavailableError(
            "The inference backend is not loaded. "
            + (f"Loading failed with: {_load_error}" if _load_error else "")
        )
    return _models


def is_loaded() -> bool:
    return _models is not None


def unload() -> None:
    """Drops the reference. Used by tests and by the retraining promotion path."""
    global _models
    with _lock:
        _models = None


def describe() -> dict:
    """Identity of what is currently loaded, for /readyz and the admin page."""
    if _models is None:
        return {"loaded": False, "error": str(_load_error) if _load_error else None}
    return {
        "loaded": True,
        "backend": _models["backend"],
        "artifact_uri": _models.get("artifact_uri"),
        "category_labels": list(_models["category_labels"]),
        "urgency_labels": list(_models["urgency_labels"]),
    }


def artifact_fingerprint(root: Path | None = None) -> str:
    """A stable digest of the weight files on disk.

    Used as the default `version` when registering a model_versions row
    for weights that arrived out-of-band -- models/ is gitignored, so
    there is otherwise nothing tying a running system to a known artefact.
    Hashes file sizes and mtimes rather than contents: a 676 MB checksum
    on every startup would be its own problem.
    """
    root = Path(root) if root is not None else get_settings().models_dir
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".json"}:
            stat = path.stat()
            digest.update(str(path.relative_to(root)).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(int(stat.st_mtime)).encode())
    return digest.hexdigest()[:16]


def read_label_classes(root: Path | None = None) -> dict[str, list[str]]:
    """Department and urgency label lists, read without loading any weights.

    Seeding the departments table needs the label list; making that wait
    on a gigabyte of transformer weights would be absurd, so the JSON is
    read directly.
    """
    root = Path(root) if root is not None else get_settings().models_dir
    labels: dict[str, list[str]] = {}

    multitask_config = root / "multitask" / "multitask_config.json"
    if multitask_config.exists():
        cfg = json.loads(multitask_config.read_text(encoding="utf-8"))
        return {
            "category": list(cfg["category_labels"]),
            "urgency": list(cfg["urgency_labels"]),
        }

    for task in ("category", "urgency"):
        directory = root / f"distilbert_{task}"
        label_file = directory / "label_classes.json"
        mapping_file = directory / "label_mapping.json"
        if label_file.exists():
            labels[task] = json.loads(label_file.read_text(encoding="utf-8"))
        elif mapping_file.exists():
            mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
            id2label = mapping.get("id2label", {})
            labels[task] = [id2label[str(i)] for i in range(len(id2label))]
        else:
            labels[task] = []
    return labels
