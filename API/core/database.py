"""SQLAlchemy engine/session + table init + seeding."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from API.core.config import settings

_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create tables if missing, then seed initial models/datasets when empty."""
    from API.models import tables  # noqa: F401 — register mappers
    Base.metadata.create_all(engine)
    from API.services.seed import seed_if_empty
    seed_if_empty()
