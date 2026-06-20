"""Model + dataset registry — DB CRUD, dict-returning interface."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import ModelRecord, DatasetRecord

ROOT = settings.ROOT
_DATA_FIELDS = ("max_nodes", "top_cwe", "filter_top25_dangerous", "max_per_class",
                "resample_seed", "storage", "ds_name_suffix")


def load_models() -> dict:
    with SessionLocal() as s:
        return {m.id: m.to_dict() for m in s.scalars(select(ModelRecord)).all()}


def load_datasets() -> dict:
    with SessionLocal() as s:
        return {d.id: d.to_dict() for d in s.scalars(select(DatasetRecord)).all()}


def get_model(model_id: str) -> dict:
    with SessionLocal() as s:
        m = s.get(ModelRecord, model_id)
        if m is None:
            raise KeyError(f"Unknown model_id '{model_id}'")
        return m.to_dict()


def get_dataset(dataset_id: str) -> dict:
    with SessionLocal() as s:
        d = s.get(DatasetRecord, dataset_id)
        if d is None:
            raise KeyError(f"Unknown dataset_id '{dataset_id}'")
        return d.to_dict()


def register_model(model_id: str, meta: dict) -> None:
    with SessionLocal() as s:
        m = s.get(ModelRecord, model_id) or ModelRecord(id=model_id)
        m.label = meta.get("label", "")
        m.arch = meta["arch"]
        m.config = meta["config"]
        m.checkpoint = meta["checkpoint"]
        m.dataset_id = meta.get("dataset_id", "")
        m.num_classes = int(meta.get("num_classes", 2))
        m.class_names = meta.get("class_names", [])
        m.base_model_id = meta.get("base_model_id")
        m.method = meta.get("method")
        s.merge(m)
        s.commit()


def register_dataset(dataset_id: str, meta: dict) -> None:
    with SessionLocal() as s:
        d = s.get(DatasetRecord, dataset_id) or DatasetRecord(id=dataset_id)
        d.label = meta.get("label", "")
        d.source = meta["source"]
        d.mode = meta.get("mode", "multiclass")
        d.num_classes = int(meta.get("num_classes", 2))
        d.params = {k: meta[k] for k in _DATA_FIELDS if k in meta}
        s.merge(d)
        s.commit()


def abspath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)
