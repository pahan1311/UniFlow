"""
settings.py
-----------
Every tunable the system reads at runtime, in one typed object.

The old app hardcoded all of this: the database path lived in db_utils,
the model directory in config, SLA targets in config, and the only
environment variables in the project were read by a hand-rolled .env
parser inside the email poller. Nothing could be pointed at a different
database or model directory without editing source.

Values resolve in the usual order: environment variable, then .env file,
then the default here. Every variable is prefixed UNIFLOW_ so the
existing .env.example keeps working unchanged.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UNIFLOW_",
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- environment ----------------------------------------------------
    env: str = Field(default="development")
    debug: bool = Field(default=False)

    # ---- storage --------------------------------------------------------
    # PostgreSQL only, e.g. postgresql+psycopg://user:password@host:5432/uniflow.
    # No default: there is no bundled fallback database, so a missing
    # UNIFLOW_DATABASE_URL is a configuration error the app should refuse
    # to start with, not silently paper over. The test suite is the one
    # exception -- it builds its own throwaway SQLite engine directly
    # (see tests/test_uniflow.py) rather than going through get_settings().
    database_url: str
    sql_echo: bool = Field(default=False)

    # ---- models ---------------------------------------------------------
    models_dir: Path = Field(default=BASE_DIR / "models")
    prefer_multitask: bool = Field(default=True)
    # Loading transformers costs ~1GB and several seconds. Tests and the
    # Streamlit client set this false so they never touch torch.
    load_models_on_startup: bool = Field(default=True)

    # ---- security -------------------------------------------------------
    secret_key: str = Field(default="dev-only-insecure-change-me")
    access_token_minutes: int = Field(default=15)
    refresh_token_days: int = Field(default=14)
    max_failed_logins: int = Field(default=5)
    lockout_minutes: int = Field(default=15)
    seed_demo_users: bool = Field(default=True)

    # ---- API / client ---------------------------------------------------
    # 5173 is Vite's dev server, reaching the API through its own proxy
    # (see frontend/vite.config.ts) as a same-origin fallback. The
    # production build is served from the API's own origin (see the
    # static-file mount in api/main.py), so CORS does not apply there.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ---- jobs -----------------------------------------------------------
    enable_scheduler: bool = Field(default=True)
    sla_sweep_seconds: int = Field(default=300)
    outbox_drain_seconds: int = Field(default=30)
    email_poll_seconds: int = Field(default=60)
    retention_days: int | None = Field(default=None)

    # ---- email intake (consumed by jobs/email_poll.py) ------------------
    imap_host: str | None = None
    imap_port: int = 993
    imap_user: str | None = None
    imap_password: str | None = None
    imap_folder: str = "INBOX"
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    send_ack: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"production", "prod"}

    @property
    def imap_enabled(self) -> bool:
        return bool(self.imap_host and self.imap_user and self.imap_password)

    @property
    def smtp_enabled(self) -> bool:
        return bool(self.send_ack and (self.smtp_host or self.imap_host))


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.is_production and settings.secret_key == "dev-only-insecure-change-me":
        raise RuntimeError(
            "UNIFLOW_SECRET_KEY must be set to a real secret when UNIFLOW_ENV=production"
        )
    return settings
