"""Health + registry listing endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from API.core.config import settings
from API.services import registry

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "device": settings.DEVICE,
            "joern_cli": settings.JOERN_CLI, "models": list(registry.load_models())}


@router.get("/models")
def list_models() -> dict:
    return registry.load_models()


@router.get("/datasets")
def list_datasets() -> dict:
    return registry.load_datasets()
