"""
analytics.py
------------
Aggregate endpoints.

Each returns tens of rows computed by the database. The page these feed
used to be built by reading the entire inquiries table into pandas on
every load and grouping it in Python.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from uniflow.api.deps import PrincipalDep, SessionDep
from uniflow.domain import analytics

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
def overview(session: SessionDep, principal: PrincipalDep) -> dict:
    return analytics.overview(session, principal)


@router.get("/departments")
def by_department(session: SessionDep, principal: PrincipalDep) -> list[dict]:
    return analytics.by_department(session, principal)


@router.get("/sla")
def sla(session: SessionDep, principal: PrincipalDep) -> dict:
    return analytics.sla_compliance(session, principal)


@router.get("/volume")
def volume(session: SessionDep, principal: PrincipalDep,
           days: int = Query(default=30, ge=1, le=365)) -> list[dict]:
    return analytics.volume_over_time(session, principal, days=days)


@router.get("/routing")
def routing(session: SessionDep, principal: PrincipalDep) -> list[dict]:
    return analytics.routing_mix(session, principal)


@router.get("/confidence")
def confidence(session: SessionDep, principal: PrincipalDep,
               buckets: int = Query(default=10, ge=2, le=50)) -> list[dict]:
    """The live-traffic confidence distribution, split by abstention.

    This is the risk-coverage trade-off measured on production data rather
    than on the frozen test split -- only possible because predictions are
    stored as rows carrying the policy that produced them.
    """
    return analytics.confidence_histogram(session, principal, buckets=buckets)


@router.get("/reviews")
def reviews(session: SessionDep, principal: PrincipalDep,
            days: int = Query(default=30, ge=1, le=365)) -> list[dict]:
    return analytics.review_activity(session, days=days)
