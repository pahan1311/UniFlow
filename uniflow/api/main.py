"""
main.py
-------
The application object.

Two things happen at startup that did not happen before.

The model is loaded exactly once, here, and every request shares it. The
old deployment held the weights twice -- once in the Streamlit process and
once in the email poller -- which is roughly 2 GB resident and the reason
the README documents a Windows allocation failure.

The scheduler starts here too. SLA escalation used to run as a side effect
of rendering a page: find_escalations() over a DataFrame followed by a
write per breached id, inside the render path. If nobody opened the
dashboard, nothing escalated.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import sqlalchemy as sa
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from uniflow import __version__
from uniflow.api.schemas import HealthOut
from uniflow.api.v1 import router as v1_router
from uniflow.core.errors import UniFlowError
from uniflow.core.settings import BASE_DIR, get_settings
from uniflow.db.session import get_engine
from uniflow.ml import registry

logger = logging.getLogger(__name__)

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )

    # No migration tool: create_all() is a no-op against an up-to-date
    # database and brings a brand-new one to a runnable (but unseeded)
    # state, so `uvicorn` alone never fails with "no such table".
    from uniflow.db.session import create_all

    create_all()

    if settings.load_models_on_startup:
        try:
            registry.load()
        except Exception:
            # A failed load must not stop the process: /readyz reports it,
            # and the queue stays readable while somebody fixes models/.
            logger.exception("continuing without an inference backend")

    scheduler = None
    if settings.enable_scheduler:
        from uniflow.jobs.scheduler import start

        scheduler = start()

    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        registry.unload()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="UniFlow API",
        version=__version__,
        summary="University inquiry routing with selective prediction",
        description=(
            "Classifies student inquiries by department and urgency, routes each to "
            "the responsible queue with a response deadline, and abstains to a human "
            "triage queue when calibrated confidence falls below the active policy's "
            "threshold. Every routing decision records the model version and the "
            "policy that produced it."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(UniFlowError)
    async def domain_error(request: Request, exc: UniFlowError):
        """Domain exceptions carry their own status code.

        This is why services can raise PermissionDeniedError without
        importing FastAPI, and stay callable from a job or a script.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, **({"context": exc.detail} if exc.detail else {})},
        )

    app.include_router(v1_router)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", response_model=HealthOut, tags=["ops"])
    def readyz(response: JSONResponse = None) -> HealthOut:
        """Ready means the database answers and the models are loaded."""
        database_ok = True
        try:
            with get_engine().connect() as connection:
                connection.execute(sa.text("SELECT 1"))
        except Exception:
            logger.exception("database health check failed")
            database_ok = False

        models = registry.describe()
        ready = database_ok and models["loaded"]
        return HealthOut(
            status="ready" if ready else "degraded",
            database=database_ok,
            models=models,
            version=__version__,
        )

    # Serves the built React app (frontend/) so this is one deployment
    # artifact and one process -- no nginx/serve/Caddy. Registered last:
    # FastAPI matches routes in registration order, so every API route
    # above still wins over the catch-all below.
    if FRONTEND_DIST.exists():
        app.mount(
            "/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets"
        )

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str) -> FileResponse:
            """Serves index.html for any path the SPA's own router owns.

            StaticFiles(html=True) alone 404s on a deep-link reload (e.g.
            reloading on /analytics), because that path has no file on
            disk -- only React Router knows what to render there. This
            route exists to hand back index.html for exactly that case.
            """
            return FileResponse(FRONTEND_DIST / "index.html")
    else:
        logger.warning(
            "%s does not exist -- run `npm run build` in frontend/ to serve the UI "
            "from this process. The API still works on its own.",
            FRONTEND_DIST,
        )

    return app


app = create_app()
