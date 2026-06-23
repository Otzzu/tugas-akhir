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


# Which config sections each kind is allowed to supply. A kind=data config serves the data-build
# spec; a kind=model config serves the architecture; kind=full (self-contained, fresh-train)
# serves anything. So a `data_config_id` accepts kind=data or full, never a model config.
_KIND_SECTIONS = {
    "data":  {"data"},
    "model": {"model"},
    "full":  {"data", "model", "train"},
}


def config_section(config_id: str, expect: str) -> dict:
    """Return the `expect` section ('data' | 'model' | 'train') of a registered config, GUARDED
    by kind: only a config whose kind is meant to carry that section may be used here. So a
    `data_config_id` must be kind=data (or full) — a kind=model config is rejected. Raises
    ValueError on a kind mismatch, KeyError on an unknown id."""
    c = get_config(config_id)
    kind = c.get("kind", "full")
    if expect not in _KIND_SECTIONS.get(kind, {"data", "model", "train"}):
        raise ValueError(
            f"config '{config_id}' (kind '{kind}') cannot supply a '{expect}' section")
    content = yaml.safe_load(c.get("content") or "") or {}
    sec = content.get(expect, {}) if isinstance(content, dict) else {}
    return sec if isinstance(sec, dict) else {}


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
    # The model's config is kind=model: architecture + train hyperparameters + the output mode,
    # WITHOUT the data-build params (source, filters, split, max_nodes). Those belong to the
    # dataset's kind=data config. Inference needs only this; relearn composes it with the task-B
    # dataset's data config at run time. (`mode` is kept under data because the library's model
    # builder + predictor read cfg.data.mode; the rest of the data block is the dataset's job.)
    src = abspath(meta["config"]) if meta.get("config") else None
    full = yaml.safe_load(src.read_text(encoding="utf-8")) if (src and src.exists()) else {}
    nm = Path(meta["config"]).stem if meta.get("config") else model_id
    model_cfg = {k: full[k] for k in ("model", "train") if isinstance(full.get(k), dict)}
    if isinstance(full.get("data"), dict) and "mode" in full["data"]:
        model_cfg["data"] = {"mode": full["data"]["mode"]}
    cfg_id = upsert_config("model", nm, model_cfg, path=meta.get("config", ""), arch=meta["arch"])
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
    cid = meta.get("data_config_id")
    if not cid:
        # seed JSON carries no config content, so synthesize a kind=data config from its build
        # params — every dataset then references one (ingested datasets already register theirs
        # in tasks.py). storage_uri is a pointer, not a build param, so it's left out.
        data_block = {k: meta[k] for k in _DATA_FIELDS if k in meta and k != "storage_uri"}
        # featurization (embedder + func_max_length) is part of how the .pt was built, so it
        # belongs in the data config's model block — relearn reads it back to match the cached .pt.
        model_block = {"num_classes": int(meta.get("num_classes", 2)), **(meta.get("featurization") or {})}
        content = {"data": {"source": meta["source"], "mode": meta.get("mode", "multiclass"),
                            **data_block},
                   "model": model_block}
        cid = upsert_config("data", meta["source"], content)
    with SessionLocal() as s:
        d = s.get(DatasetRecord, dataset_id) or DatasetRecord(id=dataset_id)
        d.label = meta.get("label", "")
        d.source = meta["source"]
        d.mode = meta.get("mode", "multiclass")
        d.num_classes = int(meta.get("num_classes", 2))
        d.data_config_id = cid
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


_CFG_CACHE = settings.API_DIR / "_config_cache"


def _require_model_config(c: dict, cid: str) -> None:
    """Guard a model's config_id — it must be a kind=model config (or full as a fresh-train
    fallback), NEVER a kind=data or kind=train config, which lack the architecture a model
    needs. Mirrors config_section's guard on the dataset/train side."""
    if c.get("kind") not in ("model", "full"):
        raise ValueError(
            f"config_id '{cid}' is kind '{c.get('kind')}', but a model needs a kind=model "
            f"(or full) config")


def require_model_config(config_id: str) -> dict:
    """Public guard for a caller-supplied config_id (e.g. /train) — fetch it and ensure it is
    usable to instantiate a model (kind=model, or full). Raises ValueError on a kind=data/train
    config, KeyError if the id is unknown."""
    c = get_config(config_id)
    _require_model_config(c, config_id)
    return c


def config_text(meta: dict) -> str:
    """Config YAML text for a model. Source of truth = the immutable DB snapshot
    (configs.content, content-addressed); falls back to the registered file path for dev
    boxes that still have the repo tree. Lets a DEPLOYED image load a model with no source
    configs baked in — the config travels with the model in the DB, not the filesystem."""
    cid = meta.get("config_id")
    if cid:
        try:
            c = get_config(cid)
        except KeyError:
            c = None
        if c is not None:
            _require_model_config(c, cid)   # raises on a kind=data/train config
            if c.get("content"):
                return c["content"]
    cfg = meta.get("config")
    if cfg and abspath(cfg).exists():
        return abspath(cfg).read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Config unavailable: no DB snapshot (config_id={cid!r}) and no file at {cfg!r}")


def materialize_config(meta: dict) -> Path:
    """Local YAML path holding a model's config — staged from the DB snapshot (cached by
    config_id, immutable) so loading needs no repo file. File fallback for dev. Mirrors how
    checkpoints/datasets materialize from object storage."""
    cid = meta.get("config_id")
    if cid:
        c = None
        try:
            c = get_config(cid)
        except KeyError:
            pass
        if c is not None:
            _require_model_config(c, cid)   # raises on a kind=data/train config
        if c and c.get("content"):
            _CFG_CACHE.mkdir(parents=True, exist_ok=True)
            f = _CFG_CACHE / f"{re.sub(r'[^0-9A-Za-z._-]+', '_', cid)}.yaml"
            if not f.exists():
                f.write_text(c["content"], encoding="utf-8")
            return f
    cfg = meta.get("config")
    if cfg and abspath(cfg).exists():
        return abspath(cfg)
    raise FileNotFoundError(
        f"Config unavailable for '{meta.get('id')}': config_id={cid!r}, file={cfg!r}")


def abspath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)
