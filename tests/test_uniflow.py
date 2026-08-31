"""
test_uniflow.py
---------------
Unit tests for the logic that does not require the transformer weights.

Deliberately excluded: anything that loads a model. Those paths are
exercised by the evaluation pipeline instead, which is both a better
test of them and produces the numbers the report needs. What is tested
here is the surrounding logic where a silent bug would corrupt results
without ever raising -- SLA arithmetic, access control, optimistic
locking, the calibration and selective-prediction mathematics, and the
review log that feeds retraining.

Every database test runs against a throwaway SQLite file created fresh
per test (the db_session fixture), completely independent of the
project's own data/uniflow.db -- nothing here touches real data.

Run with:
    python -m pytest tests/ -v
"""

import hashlib
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))                  # for the uniflow package
sys.path.insert(0, str(BASE_DIR / "evaluation"))    # calibrate/selective/dataset/stats_tests/policy_files

from uniflow.core.enums import (  # noqa: E402
    IntakeSource,
    QueueStatus,
    ReviewAction,
    Role,
    RoutingDecision,
    Urgency,
)
from uniflow.core.errors import AuthenticationError, ConflictError  # noqa: E402
from uniflow.core.security import hash_password  # noqa: E402
from uniflow.db.models import Department, Inquiry, ReviewEvent, TrainingRun, User  # noqa: E402
from uniflow.domain import auth as auth_domain  # noqa: E402
from uniflow.domain import principal as principal_module  # noqa: E402
from uniflow.domain import review, sla, workflow  # noqa: E402
from uniflow.domain.principal import Principal  # noqa: E402


# ---------------------------------------------------------------------------
# Isolated database per test
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session(tmp_path):
    """A SQLAlchemy session backed by a throwaway SQLite file.

    Independent of the global get_engine()/get_settings() singleton --
    domain functions all take a session as an explicit argument rather
    than reaching for a global connection, which is exactly what makes
    them testable like this.
    """
    from uniflow.db.base import Base
    from uniflow.db.session import _configure_sqlite  # reuse the real PRAGMAs
    import uniflow.db.models  # noqa: F401 - registers every table

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    _configure_sqlite(engine)
    Base.metadata.create_all(engine)
    session = sa.orm.sessionmaker(bind=engine, future=True)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _make_department(session, code, is_triage=False):
    dept = Department(code=code, name=code, is_triage=is_triage)
    session.add(dept)
    session.flush()
    return dept


def _make_user(session, username, role, department=None, password="secret123"):
    user = User(
        username=username, display_name=username,
        password_hash=hash_password(password), role=role,
        department_id=department.id if department else None,
    )
    session.add(user)
    session.flush()
    return user


def _make_inquiry(session, *, department=None, urgency=Urgency.LOW,
                  routing=RoutingDecision.AUTO, status=QueueStatus.NEW,
                  created_at=None, sla_due_at=None, resolved_at=None,
                  source=IntakeSource.WEB, source_ref=None, raw_text="text"):
    now = created_at or datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    inquiry = Inquiry(
        reference=f"INQ-TEST-{uuid.uuid4().hex[:8]}",
        raw_text=raw_text, normalized_text=raw_text,
        text_hash=hashlib.sha256(raw_text.encode()).hexdigest(),
        source=source, source_ref=source_ref,
        department_id=department.id if department else None,
        urgency=urgency, status=status, routing=routing,
        created_at=now, updated_at=now,
        sla_due_at=sla_due_at, resolved_at=resolved_at,
    )
    session.add(inquiry)
    session.flush()
    return inquiry


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def test_clean_text_normalises_quotes_and_whitespace():
    from uniflow.ml.preprocessing import clean_text

    assert clean_text("  I’m   stuck\n\non   Moodle ") == "I'm stuck on Moodle"
    assert clean_text(None) == ""
    assert clean_text(123) == "123"


# ---------------------------------------------------------------------------
# SLA
# ---------------------------------------------------------------------------

def test_sla_due_dates_follow_urgency(db_session):
    created = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    high = sla.due_at(db_session, Urgency.HIGH, None, created)
    low = sla.due_at(db_session, Urgency.LOW, None, created)
    assert high == datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)   # 4h target
    assert low == datetime(2026, 1, 4, 9, 0, tzinfo=timezone.utc)     # 72h target


def test_sla_state_transitions():
    """sla_state() reads plain Inquiry attributes -- no DB round-trip needed."""
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    created = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)

    def inquiry(due, resolved=None):
        return Inquiry(created_at=created, sla_due_at=due, resolved_at=resolved)

    assert sla.sla_state(inquiry(datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)),
                         now) == sla.SlaState.ON_TRACK                # 50% elapsed
    assert sla.sla_state(inquiry(datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)),
                         now) == sla.SlaState.AT_RISK                 # 80% elapsed
    assert sla.sla_state(inquiry(datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)),
                         now) == sla.SlaState.BREACHED
    # A resolved ticket is judged on when it closed, not on 'now'.
    assert sla.sla_state(
        inquiry(datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc),
               resolved=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)),
        now,
    ) == sla.SlaState.MET
    assert sla.sla_state(
        inquiry(datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
               resolved=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)),
        now,
    ) == sla.SlaState.BREACHED


def test_resolved_ticket_never_becomes_breached_by_the_passage_of_time():
    """Regression guard: compliance statistics must be stable over time."""
    inquiry = Inquiry(
        created_at=datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc),
        sla_due_at=datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc),
        resolved_at=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
    )
    for days in (0, 1, 30, 365):
        now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(days=days)
        assert sla.sla_state(inquiry, now) == sla.SlaState.MET


def test_breached_filter_matches_only_unresolved_overdue_tickets(db_session):
    """The SLA sweep's entire WHERE clause, exercised against real rows."""
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    dept = _make_department(db_session, "Admissions")

    overdue_open = _make_inquiry(
        db_session, department=dept,
        sla_due_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )
    _make_inquiry(  # already escalated -- must not match again
        db_session, department=dept,
        sla_due_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )
    db_session.execute(
        sa.update(Inquiry).where(Inquiry.id != overdue_open.id)
        .values(escalated_at=now)
    )
    _make_inquiry(  # resolved before the deadline -- must not match
        db_session, department=dept,
        sla_due_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        resolved_at=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
    )

    matched = db_session.execute(
        sa.select(Inquiry.id).where(sla.breached_filter(now))
    ).scalars().all()
    assert matched == [overdue_open.id]


def test_sla_compliance_counts_only_closed_tickets(db_session):
    from uniflow.domain import analytics

    dept = _make_department(db_session, "Admissions")
    admin = Principal.from_user(_make_user(db_session, "admin", Role.ADMIN))
    now = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)

    _make_inquiry(  # closed, met target
        db_session, department=dept,
        sla_due_at=datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc),
        resolved_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )
    _make_inquiry(  # closed, breached
        db_session, department=dept,
        sla_due_at=datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc),
        resolved_at=datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc),
    )
    _make_inquiry(  # still open, past its deadline
        db_session, department=dept,
        sla_due_at=datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc),
    )

    summary = analytics.sla_compliance(db_session, admin, now)
    assert summary["compliance_rate"] == pytest.approx(0.5)   # one of two closed met target
    resolved_late = summary["closed"] - summary["met"]
    still_open_breached = summary["open_by_state"].get("breached", 0)
    assert resolved_late + still_open_breached == 2            # one closed late, one still open


# ---------------------------------------------------------------------------
# Authentication and access control
# ---------------------------------------------------------------------------

def test_password_hashing_roundtrip():
    from uniflow.core.security import verify_password

    stored = hash_password("correct horse")
    assert verify_password("correct horse", stored)[0]
    assert not verify_password("wrong horse", stored)[0]
    assert not verify_password("correct horse", "not-a-valid-hash")[0]
    # Distinct salts mean identical passwords never share a hash.
    assert hash_password("same") != hash_password("same")


def test_legacy_pbkdf2_hash_still_verifies_and_flags_for_rehash():
    """Compatibility guarantee: existing app/auth.py hashes must keep working."""
    from uniflow.core.security import legacy_hash_password, verify_password

    legacy_stored = legacy_hash_password("correct horse")
    valid, needs_rehash = verify_password("correct horse", legacy_stored)
    assert valid
    assert needs_rehash


def test_login_succeeds_and_rejects_wrong_credentials(db_session):
    dept = _make_department(db_session, "Examinations")
    _make_user(db_session, "exams", Role.OFFICER, department=dept, password="exams123")

    pair = auth_domain.login(db_session, "exams", "exams123")
    assert pair.user.username == "exams"

    with pytest.raises(AuthenticationError):
        auth_domain.login(db_session, "exams", "wrong-password")
    with pytest.raises(AuthenticationError):
        auth_domain.login(db_session, "nobody", "whatever")


def test_login_locks_the_account_after_repeated_failures(db_session):
    from uniflow.core.settings import get_settings

    dept = _make_department(db_session, "Examinations")
    _make_user(db_session, "exams", Role.OFFICER, department=dept, password="exams123")

    for _ in range(get_settings().max_failed_logins):
        with pytest.raises(AuthenticationError):
            auth_domain.login(db_session, "exams", "wrong-password")

    with pytest.raises(AuthenticationError, match="locked"):
        auth_domain.login(db_session, "exams", "exams123")   # correct password, still locked


def test_visible_departments_scoped_to_own_plus_triage(db_session):
    triage = _make_department(db_session, "triage", is_triage=True)
    examinations = _make_department(db_session, "Examinations")
    _make_department(db_session, "Admissions")
    officer = _make_user(db_session, "exams", Role.OFFICER, department=examinations)
    admin = _make_user(db_session, "admin", Role.ADMIN)

    scope = principal_module.visible_department_ids(db_session, Principal.from_user(officer))
    assert set(scope) == {triage.id, examinations.id}

    assert principal_module.visible_department_ids(
        db_session, Principal.from_user(admin)
    ) is None   # no restriction


def test_can_read_denies_other_departments(db_session):
    triage = _make_department(db_session, "triage", is_triage=True)
    examinations = _make_department(db_session, "Examinations")
    admissions = _make_department(db_session, "Admissions")
    officer = _make_user(db_session, "exams", Role.OFFICER, department=examinations)
    principal = Principal.from_user(officer)

    own_ticket = _make_inquiry(db_session, department=examinations)
    other_ticket = _make_inquiry(db_session, department=admissions)
    triage_ticket = _make_inquiry(db_session, department=triage)

    assert principal_module.can_read(db_session, principal, own_ticket)
    assert principal_module.can_read(db_session, principal, triage_ticket)
    assert not principal_module.can_read(db_session, principal, other_ticket)


# ---------------------------------------------------------------------------
# Database, workflow and the human-in-the-loop review log
# ---------------------------------------------------------------------------

def test_new_inquiry_gets_sane_defaults(db_session):
    dept = _make_department(db_session, "Examinations")
    created = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    due = sla.due_at(db_session, Urgency.HIGH, dept.id, created_at=created)

    inquiry = _make_inquiry(db_session, department=dept, urgency=Urgency.HIGH,
                            created_at=created, sla_due_at=due)

    assert inquiry.status == QueueStatus.NEW
    assert inquiry.version == 1
    assert inquiry.sla_due_at == created + timedelta(hours=4)
    assert inquiry.resolved_at is None


def test_update_status_stamps_response_and_resolution(db_session):
    dept = _make_department(db_session, "Admissions")
    admin = Principal.from_user(_make_user(db_session, "admin", Role.ADMIN))
    inquiry = _make_inquiry(db_session, department=dept)

    workflow.update_status(db_session, admin, inquiry.id,
                           status=QueueStatus.IN_PROGRESS, expected_version=1)
    assert inquiry.first_response_at is not None
    assert inquiry.resolved_at is None
    first_response = inquiry.first_response_at

    workflow.update_status(db_session, admin, inquiry.id,
                           status=QueueStatus.RESOLVED, expected_version=2)
    assert inquiry.resolved_at is not None
    # The first-response timestamp must not move when the ticket closes.
    assert inquiry.first_response_at == first_response

    # Reopening clears the resolution time so SLA state is recomputed honestly.
    workflow.update_status(db_session, admin, inquiry.id,
                           status=QueueStatus.IN_PROGRESS, expected_version=3)
    assert inquiry.resolved_at is None


def test_stale_write_is_rejected_by_optimistic_locking(db_session):
    """The fix for the old update_queue_item lost-update race.

    Two writers holding the same starting version: the second one in
    must get a conflict, never a silent overwrite of the first.
    """
    dept = _make_department(db_session, "Admissions")
    admin = Principal.from_user(_make_user(db_session, "admin", Role.ADMIN))
    inquiry = _make_inquiry(db_session, department=dept)

    workflow.update_status(db_session, admin, inquiry.id,
                           status=QueueStatus.IN_PROGRESS, expected_version=1)
    with pytest.raises(ConflictError):
        workflow.update_status(db_session, admin, inquiry.id,
                               status=QueueStatus.RESOLVED, expected_version=1)


def test_review_events_recorded_and_consumable(db_session):
    admissions = _make_department(db_session, "Admissions")
    examinations = _make_department(db_session, "Examinations")
    admin = Principal.from_user(_make_user(db_session, "admin", Role.ADMIN))
    inquiry = _make_inquiry(db_session, department=admissions, urgency=Urgency.LOW)

    event = review.record(
        db_session, admin, inquiry.id, action=ReviewAction.CORRECT,
        corrected_department_id=examinations.id, corrected_urgency=Urgency.HIGH,
        expected_version=1,
    )
    assert event.action == ReviewAction.CORRECT
    assert inquiry.department_id == examinations.id    # correction moves the queue
    assert inquiry.urgency == Urgency.HIGH

    unconsumed = review.unconsumed(db_session)
    assert len(unconsumed) == 1
    assert unconsumed[0].id == event.id

    run = TrainingRun(kind="feedback_retrain")
    db_session.add(run)
    db_session.flush()
    review.mark_consumed(db_session, [event.id], run)

    assert review.unconsumed(db_session) == []
    # Audit log is append-only: the event is attributed, not deleted.
    assert db_session.get(ReviewEvent, event.id).consumed_by_run_id == run.id


def test_correction_rate(db_session):
    dept = _make_department(db_session, "Admissions")
    admin = Principal.from_user(_make_user(db_session, "admin", Role.ADMIN))
    inquiry = _make_inquiry(db_session, department=dept, urgency=Urgency.LOW)

    assert review.correction_rate(db_session) is None

    actions = [ReviewAction.ACCEPT, ReviewAction.ACCEPT,
              ReviewAction.CORRECT, ReviewAction.CORRECT]
    for version, action in enumerate(actions, start=1):
        kwargs = {"corrected_urgency": Urgency.HIGH} if action is ReviewAction.CORRECT else {}
        review.record(db_session, admin, inquiry.id, action=action,
                      expected_version=version, **kwargs)

    assert review.correction_rate(db_session) == pytest.approx(0.5)


def test_route_from_triage_logs_a_correction(db_session):
    triage = _make_department(db_session, "triage", is_triage=True)
    examinations = _make_department(db_session, "Examinations")
    admin = Principal.from_user(_make_user(db_session, "admin", Role.ADMIN))
    inquiry = _make_inquiry(db_session, department=triage, urgency=Urgency.LOW,
                            routing=RoutingDecision.TRIAGE)

    workflow.route(db_session, admin, inquiry.id,
                   department_id=examinations.id, expected_version=1)

    assert inquiry.department_id == examinations.id
    assert inquiry.routing == RoutingDecision.AUTO

    events = db_session.execute(
        sa.select(ReviewEvent).where(ReviewEvent.inquiry_id == inquiry.id)
    ).scalars().all()
    assert len(events) == 1
    # No prediction attached, so the outcome can only ever read as a
    # correction -- there is nothing for it to have "accepted".
    assert events[0].action == ReviewAction.CORRECT


def test_email_source_ref_uniqueness_is_enforced(db_session):
    """Replaces the old app-level dedupe guard with a test of the real
    enforcement mechanism: the UNIQUE (source, source_ref) constraint.
    """
    dept = _make_department(db_session, "Admissions")
    _make_inquiry(db_session, department=dept, source=IntakeSource.EMAIL,
                 source_ref="<msg-1@example.com>")

    with pytest.raises(sa.exc.IntegrityError):
        _make_inquiry(db_session, department=dept, source=IntakeSource.EMAIL,
                     source_ref="<msg-1@example.com>")
    db_session.rollback()


def test_create_all_is_idempotent(tmp_path):
    """create_all() replaces the old ALTER-TABLE migration loop entirely.

    There is no migration tool because there is no production data to
    preserve across a schema change -- scripts.seed.run() fully
    repopulates a fresh database. This is the property that makes that
    safe: calling create_all() against an already-current database, which
    is exactly what every application startup does, must be a no-op.
    """
    from uniflow.db.base import Base
    import uniflow.db.models  # noqa: F401

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'idempotent.db'}")
    Base.metadata.create_all(engine)
    Base.metadata.create_all(engine)   # must not raise
    assert "inquiries" in sa.inspect(engine).get_table_names()
    engine.dispose()


# ---------------------------------------------------------------------------
# Calibration mathematics
# ---------------------------------------------------------------------------

def test_temperature_scaling_preserves_predictions():
    """The core guarantee that makes calibration safe to deploy."""
    import torch

    from uniflow.ml.inference import calibrated_probs

    logits = torch.randn(64, 7, generator=torch.Generator().manual_seed(0))
    baseline = calibrated_probs(logits, 1.0).argmax(-1)
    for temperature in (0.5, 1.5, 3.0, 10.0):
        assert torch.equal(calibrated_probs(logits, temperature).argmax(-1), baseline)


def test_temperature_above_one_reduces_confidence():
    import torch

    from uniflow.ml.inference import calibrated_probs

    logits = torch.tensor([[4.0, 1.0, 0.5]])
    assert (calibrated_probs(logits, 3.0).max() < calibrated_probs(logits, 1.0).max())


def test_fit_temperature_recovers_a_known_distortion():
    """An artificially sharpened set of logits should be softened back.

    Labels are deliberately noisy. With perfect labels the temperature is
    unidentifiable -- NLL is minimised by T -> 0, since boundless
    confidence costs nothing when the model is never wrong -- which is
    the degenerate case fit_temperature clamps.
    """
    from calibrate import fit_temperature

    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(800, 5))
    labels = true_logits.argmax(axis=1)
    # Corrupt 25% of labels so the model is confidently wrong sometimes,
    # which is exactly the overconfidence temperature scaling corrects.
    noisy = rng.uniform(size=len(labels)) < 0.25
    labels[noisy] = rng.integers(0, 5, noisy.sum())

    fitted = fit_temperature(true_logits * 4.0, labels)
    assert fitted > 1.5, f"expected the fit to soften overconfidence, got T={fitted}"


def test_fit_temperature_clamps_the_degenerate_perfect_case():
    """A perfectly separable validation split must not yield T ~ 0."""
    from calibrate import MIN_TEMPERATURE, fit_temperature

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(300, 4))
    labels = logits.argmax(axis=1)            # perfectly predicted

    fitted = fit_temperature(logits * 3.0, labels)
    assert fitted >= MIN_TEMPERATURE
    # Without the clamp this drives towards zero, making every live
    # prediction report 100% confidence.
    assert fitted > 0.0


def test_expected_calibration_error_bounds():
    from calibrate import expected_calibration_error

    # Perfectly calibrated: confidence always equals observed accuracy.
    confidences = np.array([1.0] * 50)
    correct = np.array([1.0] * 50)
    error, _ = expected_calibration_error(confidences, correct)
    assert error == pytest.approx(0.0, abs=1e-9)

    # Maximally miscalibrated: certain, and always wrong.
    error, _ = expected_calibration_error(np.array([1.0] * 50), np.array([0.0] * 50))
    assert error == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------

def test_risk_coverage_curve_is_monotonic_in_coverage():
    from selective import risk_coverage_curve

    rng = np.random.default_rng(1)
    confidences = rng.uniform(0.3, 1.0, 400)
    # Correctness correlated with confidence, as a working model produces.
    correct = (rng.uniform(size=400) < confidences).astype(float)

    curve = risk_coverage_curve(confidences, correct)
    assert curve["coverage"].is_monotonic_increasing
    assert curve["coverage"].iloc[-1] == pytest.approx(1.0)
    # Full-coverage risk equals the plain error rate.
    assert curve["risk"].iloc[-1] == pytest.approx(1 - correct.mean())
    # Restricting to the most confident items must not be worse than the whole set.
    assert curve["risk"].iloc[len(curve) // 4] <= curve["risk"].iloc[-1] + 1e-9


def test_choose_tau_respects_the_target_risk():
    """With confidence ~ accuracy, a 15% target is reachable at usable coverage.

    Confidences are uniform on [0.3, 1.0] and an item is correct with
    probability equal to its confidence, so selective risk at threshold
    t is about 1 - (t+1)/2. Reaching 15% risk needs t >= 0.7, which
    still covers roughly 43% of items -- above the 30% coverage floor.
    """
    from selective import choose_tau, evaluate_at_tau, risk_coverage_curve

    rng = np.random.default_rng(2)
    confidences = rng.uniform(0.3, 1.0, 2000)
    correct = (rng.uniform(size=2000) < confidences).astype(float)

    curve = risk_coverage_curve(confidences, correct)
    tau, met = choose_tau(curve, target_risk=0.15)
    assert met
    point = evaluate_at_tau(confidences, correct, tau)
    assert point["risk"] <= 0.15 + 1e-9
    assert point["coverage"] >= 0.3
    # Abstaining must never increase the error rate among routed items.
    assert point["risk"] <= point["full_coverage_risk"] + 1e-9


def test_choose_tau_reports_failure_when_the_target_is_unreachable():
    """An unachievable target must be flagged, not silently accommodated.

    A 1% target cannot be met at 30% coverage with this distribution, and
    the caller has to know that so the report says so rather than
    presenting a threshold that does not deliver what it claims.
    """
    from selective import choose_tau, risk_coverage_curve

    rng = np.random.default_rng(2)
    confidences = rng.uniform(0.3, 1.0, 2000)
    correct = (rng.uniform(size=2000) < confidences).astype(float)

    curve = risk_coverage_curve(confidences, correct)
    _, met = choose_tau(curve, target_risk=0.01)
    assert not met


def test_evaluate_at_tau_handles_total_abstention():
    from selective import evaluate_at_tau

    point = evaluate_at_tau(np.array([0.1, 0.2]), np.array([0.0, 1.0]), tau=0.9)
    assert point["coverage"] == 0.0
    assert point["n_auto_routed"] == 0
    assert point["risk"] is None


# ---------------------------------------------------------------------------
# Energy / OOD
# ---------------------------------------------------------------------------

def test_energy_is_lower_for_confident_logits():
    import torch

    from uniflow.ml.inference import energy_score

    confident = torch.tensor([[8.0, 0.0, 0.0]])
    uncertain = torch.tensor([[0.1, 0.0, 0.05]])
    assert float(energy_score(confident)) < float(energy_score(uncertain))


def test_energy_uses_magnitude_not_just_shape():
    """The property that distinguishes energy from max-softmax.

    Two logit vectors with identical softmax outputs but different
    magnitudes must receive different energies; otherwise the detector
    adds nothing over a confidence threshold.
    """
    import torch
    import torch.nn.functional as F

    from uniflow.ml.inference import energy_score

    large = torch.tensor([[6.0, 3.0, 3.0]])
    small = large - 5.0                       # identical softmax, shifted logits
    assert torch.allclose(F.softmax(large, -1), F.softmax(small, -1), atol=1e-6)
    assert float(energy_score(large)) < float(energy_score(small))


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------

def test_mcnemar_detects_a_clear_difference():
    from stats_tests import mcnemar_exact

    gold = ["a"] * 100
    model = ["a"] * 100                       # always right
    baseline = ["a"] * 60 + ["b"] * 40        # wrong 40 times

    result = mcnemar_exact(gold, model, baseline)
    assert result["b"] == 40 and result["c"] == 0
    assert result["significant_at_05"]
    assert result["favours"] == "model"


def test_mcnemar_reports_no_difference_for_identical_systems():
    from stats_tests import mcnemar_exact

    gold = ["a", "b"] * 50
    predictions = ["a", "a"] * 50
    result = mcnemar_exact(gold, predictions, predictions)
    assert result["b"] == 0 and result["c"] == 0
    assert result["p_value"] == 1.0
    assert not result["significant_at_05"]


def test_bootstrap_interval_contains_the_point_estimate():
    from stats_tests import bootstrap_macro_f1

    rng = np.random.default_rng(3)
    gold = rng.choice(["a", "b", "c"], 200).tolist()
    predictions = [g if rng.uniform() < 0.8 else "a" for g in gold]

    result = bootstrap_macro_f1(gold, predictions, n_resamples=300)
    assert result["ci_low"] <= result["macro_f1"] <= result["ci_high"]
    assert 0.0 <= result["ci_low"] <= 1.0


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def test_column_aliases_are_resolved(tmp_path):
    from dataset import load_corpus

    csv = tmp_path / "corpus.csv"
    pd.DataFrame({
        "Student_Query": ["a question", "another question"],
        "Department": ["Examinations", "Admissions"],
        "Priority": ["High", "Low"],
    }).to_csv(csv, index=False)

    corpus = load_corpus(csv)
    assert list(corpus.columns) == ["text", "category", "urgency"]
    assert corpus.iloc[0]["category"] == "Examinations"


def test_supplied_split_is_honoured_without_a_leakage_warning(tmp_path):
    from dataset import get_splits

    csv = tmp_path / "corpus.csv"
    pd.DataFrame({
        "text": [f"question number {i}" for i in range(30)],
        "category": ["Examinations", "Admissions"] * 15,
        "urgency": ["High", "Low", "Medium"] * 10,
        "split": ["train"] * 20 + ["val"] * 5 + ["test"] * 5,
    }).to_csv(csv, index=False)

    train, val, test, leakage_risk = get_splits(csv, quiet=True)
    assert (len(train), len(val), len(test)) == (20, 5, 5)
    assert not leakage_risk


def test_generated_split_raises_the_leakage_flag(tmp_path):
    from dataset import get_splits

    csv = tmp_path / "corpus.csv"
    pd.DataFrame({
        "text": [f"question number {i}" for i in range(60)],
        "category": ["Examinations", "Admissions"] * 30,
        "urgency": ["High", "Low", "Medium"] * 20,
    }).to_csv(csv, index=False)

    train, val, test, leakage_risk = get_splits(csv, quiet=True)
    assert leakage_risk
    assert len(train) + len(val) + len(test) == 60
    # Splits must be disjoint.
    assert not set(train["text"]) & set(test["text"])
    assert not set(val["text"]) & set(test["text"])


def test_duplicate_and_empty_rows_are_dropped(tmp_path):
    from dataset import load_corpus

    csv = tmp_path / "corpus.csv"
    pd.DataFrame({
        "text": ["same question", "same question", "  ", "different"],
        "category": ["Examinations"] * 4,
        "urgency": ["High"] * 4,
    }).to_csv(csv, index=False)

    assert len(load_corpus(csv)) == 2


# ---------------------------------------------------------------------------
# Email parsing
# ---------------------------------------------------------------------------

def test_quoted_history_and_signature_are_stripped():
    import email as email_module

    from uniflow.jobs.email_intake import build_inquiry_text, extract_body

    raw = (
        "From: student@uni.edu\r\n"
        "Subject: Re: Missing result\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
        "My second semester result is still missing.\r\n"
        "\r\n"
        "On Mon, 1 Jan 2026, Registry wrote:\r\n"
        "> We are looking into it.\r\n"
        "--\r\n"
        "Sent from my phone\r\n"
    )
    message = email_module.message_from_string(raw)
    body = extract_body(message)

    assert "second semester result is still missing" in body
    assert "looking into it" not in body
    assert "Sent from my phone" not in body

    # "Re:" is stripped so the subject reads as topic, not thread metadata.
    text = build_inquiry_text("Re: Missing result", body)
    assert text.startswith("Missing result.")


def test_automated_senders_are_not_treated_as_students():
    """Regression guard: a dedicated inquiries inbox must not ingest its
    own account-security notifications as if they were student mail --
    which is exactly what happened the first time this poller ran against
    a freshly created Gmail account.
    """
    from uniflow.jobs.email_intake import is_automated_sender

    assert is_automated_sender("no-reply@accounts.google.com")
    assert is_automated_sender("notifications@github.com")
    assert is_automated_sender("MAILER-DAEMON@uni.edu")
    assert not is_automated_sender("student@uni.edu")
    assert not is_automated_sender(None)
    assert not is_automated_sender("not-an-address")


# ---------------------------------------------------------------------------
# Evaluation-pipeline policy files
# ---------------------------------------------------------------------------
# These are the file-based thresholds.json / calibration.json the offline
# evaluation pipeline reads and writes -- deliberately still file-based
# (see evaluation/policy_files.py) so `python evaluation/run_all.py` works
# before a database has ever been created. The live system reads its
# active policy from the database instead (uniflow.domain.policy).

def test_thresholds_fall_back_when_the_policy_file_is_absent(tmp_path, monkeypatch):
    import policy_files

    monkeypatch.setattr(policy_files, "THRESHOLDS_FILE", tmp_path / "missing.json")
    thresholds = policy_files.get_thresholds()
    assert thresholds["category"]["tau"] == policy_files.DEFAULT_THRESHOLDS["category"]["tau"]


def test_corrupt_policy_file_does_not_break_startup(tmp_path, monkeypatch):
    """The pipeline must still run if a policy file is truncated mid-write."""
    import policy_files

    broken = tmp_path / "broken.json"
    broken.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(policy_files, "CALIBRATION_FILE", broken)
    assert policy_files.get_calibration()["category"]["temperature"] == 1.0


def test_partial_policy_file_is_merged_over_defaults(tmp_path, monkeypatch):
    import policy_files

    partial = tmp_path / "partial.json"
    partial.write_text('{"category": {"tau": 0.83}}', encoding="utf-8")
    monkeypatch.setattr(policy_files, "THRESHOLDS_FILE", partial)

    thresholds = policy_files.get_thresholds()
    assert thresholds["category"]["tau"] == 0.83
    # The untouched task keeps its default rather than disappearing.
    assert "tau" in thresholds["urgency"]
