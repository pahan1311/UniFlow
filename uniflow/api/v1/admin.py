"""
admin.py
--------
Reference data, and the ML control plane.

Promoting a model and activating a routing policy are the two operations
that change how every future inquiry is routed. They used to be a
shutil.copytree over a live directory and a JSON file overwrite
respectively -- no record of who did it, when, or what was replaced.
"""

from __future__ import annotations

import sqlalchemy as sa
from fastapi import APIRouter

from uniflow.api.deps import AdminDep, PrincipalDep, SessionDep
from uniflow.api.schemas import (
    DepartmentOut,
    ModelVersionOut,
    RoutingPolicyCreate,
    RoutingPolicyOut,
    UserOut,
)
from uniflow.db.models import Department, ModelVersion, RoutingPolicy, User
from uniflow.domain import policy as policy_service
from uniflow.domain.principal import visible_department_ids

router = APIRouter(tags=["admin"])


@router.get("/departments", response_model=list[DepartmentOut])
def departments(session: SessionDep, principal: PrincipalDep, all_departments: bool = False):
    """Departments the caller may see.

    all_departments=true returns the full list, which the triage screen
    needs so an officer can route an unplaced inquiry to a queue they
    themselves cannot read.
    """
    query = sa.select(Department).where(Department.is_active.is_(True))
    if not all_departments:
        allowed = visible_department_ids(session, principal)
        if allowed is not None:
            query = query.where(Department.id.in_(allowed or [-1]))
    return [
        DepartmentOut.model_validate(row)
        for row in session.execute(query.order_by(Department.name)).scalars()
    ]


@router.get("/users", response_model=list[UserOut])
def users(session: SessionDep, principal: PrincipalDep):
    """Assignable colleagues.

    Scoped to the caller's own department so the assignee picker cannot be
    used to enumerate the whole staff directory.
    """
    query = sa.select(User).where(User.is_active.is_(True))
    if not principal.is_admin:
        query = query.where(User.department_id == principal.department_id)
    return [
        UserOut.model_validate(row)
        for row in session.execute(query.order_by(User.display_name)).scalars()
    ]


@router.get("/models", response_model=list[ModelVersionOut])
def models(session: SessionDep, principal: PrincipalDep):
    return [
        ModelVersionOut.model_validate(row)
        for row in session.execute(
            sa.select(ModelVersion).order_by(
                ModelVersion.name, ModelVersion.created_at.desc()
            )
        ).scalars()
    ]


@router.post("/models/{model_version_id}/promote", response_model=ModelVersionOut)
def promote(model_version_id: int, session: SessionDep, principal: AdminDep):
    version = policy_service.promote_model(session, model_version_id, principal.user_id)
    return ModelVersionOut.model_validate(version)


@router.get("/policies", response_model=list[RoutingPolicyOut])
def policies(session: SessionDep, principal: PrincipalDep):
    return [
        RoutingPolicyOut.model_validate(row)
        for row in session.execute(
            sa.select(RoutingPolicy).order_by(RoutingPolicy.created_at.desc())
        ).scalars()
    ]


@router.post("/policies", response_model=RoutingPolicyOut, status_code=201)
def create_policy(payload: RoutingPolicyCreate, session: SessionDep, principal: AdminDep):
    """Registers a policy produced by the evaluation pipeline.

    evaluation/calibrate.py and evaluation/selective.py post here instead
    of overwriting models/calibration.json and models/thresholds.json, so
    the numbers arrive with provenance and are not silently swapped under
    a running system.
    """
    policy = RoutingPolicy(
        version=payload.version,
        category_tau=payload.category_tau,
        category_temperature=payload.category_temperature,
        category_energy_threshold=payload.category_energy_threshold,
        urgency_tau=payload.urgency_tau,
        urgency_temperature=payload.urgency_temperature,
        duplicate_threshold=payload.duplicate_threshold,
        duplicate_lookback_days=payload.duplicate_lookback_days,
        derived_from=payload.derived_from,
    )
    session.add(policy)
    session.flush()
    if payload.activate:
        policy_service.activate_policy(session, policy.id, principal.user_id)
    return RoutingPolicyOut.model_validate(policy)


@router.post("/policies/{policy_id}/activate", response_model=RoutingPolicyOut)
def activate_policy(policy_id: int, session: SessionDep, principal: AdminDep):
    policy = policy_service.activate_policy(session, policy_id, principal.user_id)
    return RoutingPolicyOut.model_validate(policy)
