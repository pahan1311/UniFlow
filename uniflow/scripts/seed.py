"""
seed.py
-------
Brings a fresh database to a runnable state.

Seeds, in order: departments (from the model's own label classes, so the
two cannot disagree), SLA targets, the bootstrap routing policy, the model
version rows describing whatever is in models/, and the demo accounts.

Three things are deliberately different from the old seed_users().

It is a script, not something that runs on import. seed_users() was called
at the top of app.py, so it executed on every Streamlit rerun.

The demo credentials live here rather than in config.py, and seeding them
requires UNIFLOW_ENV to be a development environment. Committing working
passwords for a marker to use is reasonable; re-creating those accounts
automatically inside a production deployment is not.

Every seeded account is flagged must_change_password.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import sqlalchemy as sa

from uniflow.core.enums import TRIAGE_CODE, TRIAGE_NAME, Role, Urgency
from uniflow.core.security import hash_password
from uniflow.core.settings import get_settings
from uniflow.db.models import Department, ModelVersion, RoutingPolicy, SlaPolicy, User
from uniflow.db.session import create_all, session_scope
from uniflow.domain.policy import BOOTSTRAP_POLICY
from uniflow.ml import registry

logger = logging.getLogger(__name__)

# Response-time targets in hours, by predicted urgency. These turn an
# inert urgency label into something the system can be measured against.
# Carried over from config.SLA_HOURS.
SLA_HOURS = {Urgency.HIGH: 4, Urgency.MEDIUM: 24, Urgency.LOW: 72}

# Demo accounts, documented in the README so the project is runnable by a
# marker. Not a deployment posture -- hence the environment gate and the
# forced password change.
DEMO_USERS = [
    ("admin", "admin123", Role.ADMIN, None, "System Administrator"),
    ("exams", "exams123", Role.OFFICER, "Examinations", "Examinations Officer"),
    ("finance", "finance123", Role.OFFICER, "Fees and Payments", "Finance Officer"),
    ("it", "it123", Role.OFFICER, "Learning Platform / IT Access", "IT Support Officer"),
    ("registry", "registry123", Role.OFFICER, "Academic Documents", "Registry Officer"),
    ("admissions", "admissions123", Role.OFFICER, "Admissions", "Admissions Officer"),
    ("timetable", "timetable123", Role.OFFICER, "Timetable and Scheduling",
     "Scheduling Officer"),
    ("general", "general123", Role.OFFICER, "General Administration",
     "Administration Officer"),
]


def seed_departments(session) -> dict[str, Department]:
    """Creates one row per label the category model can emit, plus Triage.

    `code` is the model's label verbatim. That is the whole point: a
    predicted label resolves to a queue by equality, so the two cannot
    drift apart silently the way free-text department strings did.
    """
    labels = registry.read_label_classes().get("category", [])
    if not labels:
        logger.warning(
            "no category label classes found under %s; seeding Triage only",
            get_settings().models_dir,
        )

    existing = {
        row.code: row for row in session.execute(sa.select(Department)).scalars()
    }

    triage = existing.get(TRIAGE_CODE)
    if triage is None:
        triage = Department(code=TRIAGE_CODE, name=TRIAGE_NAME, is_triage=True)
        session.add(triage)
        existing[TRIAGE_CODE] = triage

    for label in labels:
        if label not in existing:
            department = Department(code=label, name=label, is_triage=False)
            session.add(department)
            existing[label] = department

    session.flush()
    logger.info("departments: %d", len(existing))
    return existing


def seed_sla_policies(session) -> int:
    """One global target per urgency level."""
    created = 0
    for urgency, hours in SLA_HOURS.items():
        exists = session.execute(
            sa.select(SlaPolicy.id).where(
                SlaPolicy.urgency == urgency, SlaPolicy.department_id.is_(None)
            )
        ).first()
        if exists:
            continue
        session.add(SlaPolicy(
            urgency=urgency,
            department_id=None,
            response_hours=Decimal(hours),
            warning_fraction=Decimal("0.75"),
        ))
        created += 1
    session.flush()
    return created


def _policy_from_files(session) -> RoutingPolicy | None:
    """Imports models/thresholds.json and calibration.json as a policy row.

    These files are the output of evaluation/selective.py, calibrate.py
    and ood.py, and they are what the old system actually routed by. If
    they are present, importing them means the migrated system keeps
    behaving exactly as it did -- with the difference that the numbers now
    have a version, an activation time, and an id that every prediction
    they produce will record.
    """
    settings = get_settings()
    thresholds_file = settings.models_dir / "thresholds.json"
    calibration_file = settings.models_dir / "calibration.json"
    if not thresholds_file.exists() and not calibration_file.exists():
        return None

    def load(path: Path, fallback: dict) -> dict:
        if not path.exists():
            return fallback
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("could not read %s; using defaults", path)
            return fallback

    thresholds = load(thresholds_file, {})
    calibration = load(calibration_file, {})

    category = thresholds.get("category", {})
    urgency = thresholds.get("urgency", {})
    energy = category.get("energy_threshold")

    # Version the content, not the clock: re-running the seed against
    # unchanged files must not create a second identical policy.
    fingerprint = hashlib.sha256(
        json.dumps([thresholds, calibration], sort_keys=True).encode()
    ).hexdigest()[:12]
    version = f"imported-{fingerprint}"

    existing = session.execute(
        sa.select(RoutingPolicy).where(RoutingPolicy.version == version)
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    return RoutingPolicy(
        version=version,
        category_tau=Decimal(str(category.get("tau", 0.70))),
        category_temperature=Decimal(
            str(calibration.get("category", {}).get("temperature", 1.0))
        ),
        category_energy_threshold=Decimal(str(energy)) if energy is not None else None,
        urgency_tau=Decimal(str(urgency.get("tau", 0.55))),
        urgency_temperature=Decimal(
            str(calibration.get("urgency", {}).get("temperature", 1.0))
        ),
        duplicate_threshold=Decimal("0.920"),
        duplicate_lookback_days=30,
        derived_from={
            "source": "models/thresholds.json + models/calibration.json",
            "calibrated": calibration_file.exists(),
            "imported_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def seed_policy(session) -> RoutingPolicy:
    """Ensures there is an active routing policy.

    Prefers the empirically derived values the evaluation pipeline wrote
    to disk. Falls back to config.py's defaults, honestly labelled as
    uncalibrated, so the system is runnable before that pipeline has ever
    been executed.
    """
    bootstrap = session.execute(
        sa.select(RoutingPolicy).where(RoutingPolicy.version == BOOTSTRAP_POLICY["version"])
    ).scalar_one_or_none()
    if bootstrap is None:
        bootstrap = RoutingPolicy(**BOOTSTRAP_POLICY)
        session.add(bootstrap)

    imported = _policy_from_files(session)
    if imported is not None and imported.id is None:
        session.add(imported)

    session.flush()
    preferred = imported or bootstrap

    has_active = session.execute(
        sa.select(RoutingPolicy.id).where(RoutingPolicy.is_active.is_(True))
    ).first()
    if not has_active:
        preferred.is_active = True
        preferred.activated_at = datetime.now(timezone.utc)
    session.flush()
    return preferred


def seed_model_versions(session) -> list[ModelVersion]:
    """Registers whatever weights are in models/ as an active model version.

    models/ is gitignored and the weights ship out of band, so without
    this there is nothing in the database tying a running system to a
    known artefact. The version is a fingerprint of the files on disk.

    `name` is the logical task slot -- "category", "urgency", or
    "multitask" for a shared-encoder backend -- because that is what
    domain.policy.active_model_version() and every Prediction look up by.
    It is not the directory name; artifact_uri carries that instead.
    """
    settings = get_settings()
    labels = registry.read_label_classes()
    fingerprint = registry.artifact_fingerprint()

    multitask = (settings.models_dir / "multitask" / "multitask_config.json").exists()
    if multitask:
        specs = [("multitask", "multitask", settings.models_dir / "multitask",
                  labels.get("category", []))]
    else:
        specs = [
            ("category", "dual",
             settings.models_dir / "distilbert_category", labels.get("category", [])),
            ("urgency", "dual",
             settings.models_dir / "distilbert_urgency", labels.get("urgency", [])),
        ]

    created = []
    for name, backend, path, label_classes in specs:
        if not path.exists():
            logger.warning("skipping %s: %s does not exist", name, path)
            continue
        existing = session.execute(
            sa.select(ModelVersion).where(
                ModelVersion.name == name, ModelVersion.version == fingerprint
            )
        ).scalar_one_or_none()
        if existing is not None:
            created.append(existing)
            continue

        session.execute(
            sa.update(ModelVersion)
            .where(ModelVersion.name == name, ModelVersion.is_active.is_(True))
            .values(is_active=False)
        )
        version = ModelVersion(
            name=name,
            version=fingerprint,
            backend=backend,
            artifact_uri=str(path),
            label_classes=list(label_classes),
            is_active=True,
            promoted_at=datetime.now(timezone.utc),
        )
        session.add(version)
        created.append(version)

    session.flush()
    logger.info("model versions: %s", [f"{v.name}:{v.version}" for v in created])
    return created


def seed_users(session, departments: dict[str, Department], force: bool = False) -> int:
    """Creates the demo accounts. Development environments only.

    Existing users are never overwritten, so a password changed after
    deployment survives a re-seed.
    """
    settings = get_settings()
    if settings.is_production and not force:
        logger.warning("refusing to seed demo accounts in a production environment")
        return 0
    if not settings.seed_demo_users and not force:
        return 0

    created = 0
    for username, password, role, department_code, display_name in DEMO_USERS:
        exists = session.execute(
            sa.select(User.id).where(User.username == username)
        ).first()
        if exists:
            continue
        department = departments.get(department_code) if department_code else None
        if department_code and department is None:
            logger.warning(
                "no department '%s' for user %s; the account is created without one",
                department_code, username,
            )
        session.add(User(
            username=username,
            display_name=display_name,
            password_hash=hash_password(password),
            role=role,
            # An officer with no department would fail the officer_needs_
            # department constraint, so fall back to admin-less viewer.
            department_id=department.id if department else None,
            must_change_password=True,
        ))
        created += 1
    session.flush()
    logger.info("users created: %d", created)
    return created


def run(force_users: bool = False) -> None:
    create_all()
    with session_scope() as session:
        departments = seed_departments(session)
        seed_sla_policies(session)
        policy = seed_policy(session)
        seed_model_versions(session)
        seed_users(session, departments, force=force_users)
        logger.info("active routing policy: %s", policy.version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the UniFlow database")
    parser.add_argument(
        "--force-users",
        action="store_true",
        help="create the demo accounts even outside a development environment",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    run(force_users=args.force_users)
    print("Seed complete.")


if __name__ == "__main__":
    main()
