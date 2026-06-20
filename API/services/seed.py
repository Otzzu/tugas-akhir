"""Seed the DB from API/seeds/*.json on first init (idempotent)."""
from __future__ import annotations

import json

from sqlalchemy import select, func

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import ModelRecord, DatasetRecord
from API.services import registry

MODELS_SEED = settings.SEEDS_DIR / "models.json"
DATASETS_SEED = settings.SEEDS_DIR / "datasets.json"


def seed_if_empty() -> None:
    with SessionLocal() as s:
        n_models = s.scalar(select(func.count()).select_from(ModelRecord))
        n_datasets = s.scalar(select(func.count()).select_from(DatasetRecord))
    if n_datasets == 0 and DATASETS_SEED.exists():
        for did, meta in json.loads(DATASETS_SEED.read_text()).items():
            registry.register_dataset(did, meta)
    if n_models == 0 and MODELS_SEED.exists():
        for mid, meta in json.loads(MODELS_SEED.read_text()).items():
            registry.register_model(mid, meta)
