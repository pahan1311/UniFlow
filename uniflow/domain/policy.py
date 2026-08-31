"""
policy.py
---------
Resolving the active model version and routing policy.

The old control plane was two JSON files. evaluation/calibrate.py and
evaluation/selective.py wrote them, config.get_thresholds() re-read them
on every page load, and nothing recorded which values had produced a
given decision -- so a stored routing decision could not be reproduced
once the files had been overwritten.

Here the active policy is a row, resolved once per request, and its id is
stamped onto every prediction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import sqlalchemy as sa

from uniflow.core.errors import ConflictError, NotFoundError
from uniflow.db.models import ModelVersion, RoutingPolicy

# The values config.py used before the evaluation pipeline had ever run.
# Kept as the content of the seeded "bootstrap" policy so the system is
# runnable, and honestly labelled as uncalibrated, from a cold start.
BOOTSTRAP_POLICY = {
    "version": "bootstrap",
    "category_tau": Decimal("0.70000"),
    "category_temperature": Decimal("1.0000"),
    "category_energy_threshold": None,
    "urgency_tau": Decimal("0.55000"),
    "urgency_temperature": Decimal("1.0000"),
    "duplicate_threshold": Decimal("0.920"),
    "duplicate_lookback_days": 30,
    "derived_from": {"source": "config.py defaults", "calibrated": False},
}


def active_policy(session) -> RoutingPolicy:
    policy = session.execute(
        sa.select(RoutingPolicy).where(RoutingPolicy.is_active.is_(True))
    ).scalar_one_or_none()
    if policy is None:
        raise NotFoundError(
            "No active routing policy. Run `uniflow-admin seed` or activate one "
            "via POST /api/v1/policies/{id}/activate."
        )
    return policy


def active_model_version(session, name: str = "category") -> ModelVersion:
    """The model version currently serving predictions.

    `name` is the logical model slot. The dual backend registers one row
    per task; the multitask backend registers a single 'multitask' row and
    both slots resolve to it.
    """
    version = session.execute(
        sa.select(ModelVersion)
        .where(ModelVersion.is_active.is_(True), ModelVersion.name == name)
    ).scalar_one_or_none()
    if version is None:
        version = session.execute(
            sa.select(ModelVersion)
            .where(ModelVersion.is_active.is_(True), ModelVersion.name == "multitask")
        ).scalar_one_or_none()
    if version is None:
        raise NotFoundError(
            f"No active model version for '{name}'. Register the weights in "
            "models/ with `uniflow-admin seed`."
        )
    return version


def activate_policy(session, policy_id: int, actor_id: int | None) -> RoutingPolicy:
    """Makes one policy the active one, atomically.

    Deactivating and activating in a single transaction is the difference
    between this and the old file swap, which could leave thresholds.json
    and calibration.json describing two different experiments if it was
    interrupted between writes.
    """
    policy = session.get(RoutingPolicy, policy_id)
    if policy is None:
        raise NotFoundError(f"Routing policy {policy_id} does not exist")

    session.execute(
        sa.update(RoutingPolicy)
        .where(RoutingPolicy.is_active.is_(True))
        .values(is_active=False)
    )
    policy.is_active = True
    policy.activated_at = datetime.now(timezone.utc)
    policy.activated_by_id = actor_id
    session.flush()
    return policy


def promote_model(session, model_version_id: int, actor_id: int | None) -> ModelVersion:
    """Makes one model version active for its name slot.

    The old app promoted a retrained model with shutil.copytree over a
    live directory, with no way to fail safely and no record of who
    promoted what. This is the transactional, audited equivalent.
    """
    version = session.get(ModelVersion, model_version_id)
    if version is None:
        raise NotFoundError(f"Model version {model_version_id} does not exist")
    if not version.label_classes:
        raise ConflictError("Refusing to promote a model version with no label classes")

    session.execute(
        sa.update(ModelVersion)
        .where(ModelVersion.name == version.name, ModelVersion.is_active.is_(True))
        .values(is_active=False)
    )
    version.is_active = True
    version.promoted_at = datetime.now(timezone.utc)
    version.promoted_by_id = actor_id
    session.flush()
    return version
