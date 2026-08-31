"""
policy_files.py
----------------
Reads and writes the routing-policy JSON files the evaluation pipeline
produces: models/thresholds.json and models/calibration.json.

The live system stores its active policy in the database
(uniflow.db.models.RoutingPolicy) rather than reading these files --
uniflow/scripts/seed.py imports whichever of them exists at seed time.
But the evaluation pipeline is a separate, standalone artefact-producing
process and stays file-based deliberately: `python evaluation/run_all.py`
must keep working before a database has ever been created, since it is
what produces the numbers the database gets seeded from in the first
place.

This is the same subset of the old app/config.py that the evaluation
scripts actually used -- just the JSON policy I/O, not the whole module.
"""

import json
from pathlib import Path

from uniflow.core.settings import get_settings

MODELS_DIR = get_settings().models_dir
THRESHOLDS_FILE = MODELS_DIR / "thresholds.json"
CALIBRATION_FILE = MODELS_DIR / "calibration.json"

DEFAULT_THRESHOLDS = {
    "category": {"tau": 0.70, "energy_threshold": None},
    "urgency": {"tau": 0.55},
}
DEFAULT_CALIBRATION = {
    "category": {"temperature": 1.0},
    "urgency": {"temperature": 1.0},
}


def _load_json_or(path: Path, fallback: dict) -> dict:
    """Reads a JSON policy file, falling back to the hard-coded default.

    Written so the pipeline runs correctly *before* calibrate.py or
    selective.py has ever been executed -- it simply behaves as an
    uncalibrated system until you produce the real numbers.
    """
    if not path.exists():
        return json.loads(json.dumps(fallback))
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError):
        return json.loads(json.dumps(fallback))

    merged = json.loads(json.dumps(fallback))
    for task, values in loaded.items():
        if isinstance(values, dict):
            merged.setdefault(task, {}).update(values)
    return merged


def get_thresholds() -> dict:
    return _load_json_or(THRESHOLDS_FILE, DEFAULT_THRESHOLDS)


def get_calibration() -> dict:
    return _load_json_or(CALIBRATION_FILE, DEFAULT_CALIBRATION)
