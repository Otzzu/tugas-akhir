"""Health + registry listing endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from API.core.config import settings
from API.services import registry

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict:
    checks = {"database": _check_database(), "storage": _check_storage()}
    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "device": settings.DEVICE,
            "joern_cli": settings.JOERN_CLI, "storage": settings.STORAGE_BACKEND,
            "models": list(registry.load_models()), "checks": checks}


def _check_database() -> str:
    try:
        from API.core.database import SessionLocal
        from sqlalchemy import text
        with SessionLocal() as s:
            s.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "error"


def _check_storage() -> str:
    try:
        from API.services import storage
        storage.exists(settings.S3_BUCKET_GRAPHS, "__healthcheck__")
        return "ok"
    except Exception:
        return "error"


@router.get("/models")
def list_models() -> dict:
    return registry.load_models()


@router.get("/datasets")
def list_datasets() -> dict:
    return registry.load_datasets()


@router.get("/configs")
def list_configs(kind: str | None = None) -> dict:
    """All configs, or only one kind (data | model | train | full) via ?kind=."""
    return registry.list_configs(kind)


@router.get("/configs/{config_id}")
def get_config(config_id: str) -> dict:
    try:
        return registry.get_config(config_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
