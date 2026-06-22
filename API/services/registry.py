"""Model + dataset + config registry — DB CRUD, dict-returning interface.

Everything here is APPEND-ONLY / content-addressed: configs are immutable
(`{kind}:{name}@{hash}`), datasets and models get a fresh id per build/run. Editing
never overwrites a prior id — a new id is minted, so any reference stays reproducible."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from sqlalchemy import select

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import ModelRecord, DatasetRecord, ConfigRecord, ModelArtifact

ROOT = settings.ROOT
_DATA_FIELDS = ("max_nodes", "top_cwe", "filter_top25_dangerous", "max_per_class",
                "resample_seed", "storage", "ds_name_suffix", "storage_uri",
                "val_fraction", "test_fraction")


def _config_slug(path: str) -> str:
    """Stable, unique config id derived from the (repo-relative) config path."""
    p = Path(path)
    try:
        rel = p.relative_to(ROOT) if p.is_absolute() else p
    except ValueError:
        rel = p
    return re.sub(r"[^0-9A-Za-z]+", "_", rel.with_suffix("").as_posix()).strip("_").lower()


def load_models() -> dict:
    with SessionLocal() as s:
        return {m.id: m.to_dict() for m in s.scalars(select(ModelRecord)).all()}


def load_datasets() -> dict:
    with SessionLocal() as s:
        return {d.id: d.to_dict() for d in s.scalars(select(DatasetRecord)).all()}


def load_configs() -> dict:
    with SessionLocal() as s:
        return {c.id: c.to_dict() for c in s.scalars(select(ConfigRecord)).all()}


def get_config(config_id: str) -> dict:
    with SessionLocal() as s:
        c = s.get(ConfigRecord, config_id)
        if c is None:
            raise KeyError(f"Unknown config_id '{config_id}'")
        return c.to_dict()


def list_configs(kind: str | None = None) -> dict:
    with SessionLocal() as s:
        q = select(ConfigRecord)
        if kind:
            q = q.where(ConfigRecord.kind == kind)
        return {c.id: c.to_dict() for c in s.scalars(q).all()}


def _canonical(content) -> str:
    """Stable YAML text so logically-equal configs hash identically (sorted keys)."""
    obj = yaml.safe_load(content) if isinstance(content, str) else content
    return yaml.safe_dump(obj or {}, sort_keys=True, default_flow_style=False)


def upsert_config(kind: str, name: str, content, *, path: str = "", arch: str = "") -> str:
    """Content-addressed, IMMUTABLE config registration.

    id = '{kind}:{name}@{hash}'. Same content → same id (dedup, no write). Different
    content → new id; the old row is never mutated. kind ∈ data | model | train | full."""
    canon = _canonical(content)
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:10]
    cfg_id = f"{kind}:{name}@{h}"
    with SessionLocal() as s:
        if s.get(ConfigRecord, cfg_id) is None:
            s.add(ConfigRecord(id=cfg_id, kind=kind, name=name, arch=arch,
                               path=path, content=canon, content_hash=h))
            s.commit()
    return cfg_id


def register_config(meta: dict) -> str:
    """Snapshot a config (from {path} or {content}) as an immutable content-addressed row.
    kind defaults to 'full' (a monolithic data+model+train file). Returns config_id."""
    path = meta.get("path", "")
    content = meta.get("content")
    if content is None and path:
        fp = abspath(path)
        content = fp.read_text(encoding="utf-8") if fp.exists() else ""
    name = meta.get("name") or (Path(path).stem if path else "config")
    return upsert_config(meta.get("kind", "full"), name, content or "",
                         path=path, arch=meta.get("arch", ""))


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
    cfg_id = register_config({"path": meta["config"], "arch": meta["arch"],
                              "id": meta.get("config_id")})
    with SessionLocal() as s:
        m = s.get(ModelRecord, model_id) or ModelRecord(id=model_id)
        m.label = meta.get("label", "")
        m.arch = meta["arch"]
        m.config = meta["config"]
        m.config_id = cfg_id
        m.checkpoint = meta["checkpoint"]
        m.storage_uri = meta.get("storage_uri")
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
        d.data_config_id = meta.get("data_config_id")
        d.params = {k: meta[k] for k in _DATA_FIELDS if k in meta}
        s.merge(d)
        s.commit()


def add_artifact(model_id: str, kind: str, storage_uri: str, meta: dict | None = None) -> None:
    """UPSERT a replaceable per-model artifact by (model_id, kind)."""
    with SessionLocal() as s:
        a = s.scalar(select(ModelArtifact).where(
            ModelArtifact.model_id == model_id, ModelArtifact.kind == kind))
        if a is None:
            a = ModelArtifact(model_id=model_id, kind=kind)
        a.storage_uri = storage_uri
        a.meta = meta or {}
        s.add(a)
        s.commit()


def get_artifact(model_id: str, kind: str) -> dict | None:
    with SessionLocal() as s:
        a = s.scalar(select(ModelArtifact).where(
            ModelArtifact.model_id == model_id, ModelArtifact.kind == kind))
        return a.to_dict() if a else None


def list_artifacts(model_id: str | None = None) -> list[dict]:
    with SessionLocal() as s:
        q = select(ModelArtifact)
        if model_id:
            q = q.where(ModelArtifact.model_id == model_id)
        return [a.to_dict() for a in s.scalars(q).all()]


def abspath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)
