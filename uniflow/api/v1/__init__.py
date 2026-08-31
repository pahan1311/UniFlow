"""Version 1 of the HTTP API."""

from fastapi import APIRouter

from uniflow.api.v1 import admin, analytics, auth, inquiries

router = APIRouter(prefix="/api/v1")
router.include_router(auth.router)
router.include_router(inquiries.router)
router.include_router(analytics.router)
router.include_router(admin.router)

__all__ = ["router"]
