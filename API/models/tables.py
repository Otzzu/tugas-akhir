"""SQLAlchemy ORM tables: models, datasets, configs, relearn_jobs, graph_cache."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import String, Integer, Text, JSON, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from API.core.database import Base


class ConfigRecord(Base):
    """Content-addressed, IMMUTABLE config. id = '{kind}:{name}@{hash}'. Editing a config
    never rewrites a row — same content reuses the id (dedup), different content gets a new
    id, mirroring how datasets/models are append-only. `kind` distinguishes the type so a
    caller (and each endpoint) knows what it is. `content` is the canonical YAML snapshot."""
    __tablename__ = "configs"
    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), default="full")   # data | model | train | full
    name: Mapped[str] = mapped_column(String(256), default="")
    arch: Mapped[str] = mapped_column(String(64), default="")
    path: Mapped[str] = mapped_column(Text, default="")             # original source path (informational)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name, "arch": self.arch,
                "path": self.path, "content": self.content, "content_hash": self.content_hash}


class ModelRecord(Base):
    __tablename__ = "models"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(256), default="")
    arch: Mapped[str] = mapped_column(String(64))
    config: Mapped[str] = mapped_column(Text)
    config_id: Mapped[str | None] = mapped_column(ForeignKey("configs.id"), nullable=True)
    checkpoint: Mapped[str] = mapped_column(Text)                                  # local path (cache)
    storage_uri: Mapped[str | None] = mapped_column(Text, nullable=True)           # -> checkpoints bucket (source of truth)
    dataset_id: Mapped[str] = mapped_column(String(128), default="")
    num_classes: Mapped[int] = mapped_column(Integer, default=2)
    class_names: Mapped[list] = mapped_column(JSON, default=list)
    base_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label, "arch": self.arch, "config": self.config,
            "config_id": self.config_id,
            "checkpoint": self.checkpoint, "storage_uri": self.storage_uri, "dataset_id": self.dataset_id,
            "num_classes": self.num_classes, "class_names": self.class_names or [],
            "base_model_id": self.base_model_id, "method": self.method,
        }


class ModelArtifact(Base):
    """Generic per-model artifact pointer (first use: EWC importance). One row per
    (model_id, kind); the blob lives in object storage, this keeps the storage_uri."""
    __tablename__ = "model_artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)   # -> models.id
    kind: Mapped[str] = mapped_column(String(32))                    # ewc_importance | future kinds
    storage_uri: Mapped[str] = mapped_column(Text)                   # s3://checkpoints/<...>
    meta: Mapped[dict | None] = mapped_column(JSON, default=dict, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {"id": self.id, "model_id": self.model_id, "kind": self.kind,
                "storage_uri": self.storage_uri, "meta": self.meta or {},
                "created_at": self.created_at.isoformat() if self.created_at else None}


class DatasetRecord(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(256), default="")
    source: Mapped[str] = mapped_column(String(128))
    mode: Mapped[str] = mapped_column(String(32), default="multiclass")
    num_classes: Mapped[int] = mapped_column(Integer, default=2)
    data_config_id: Mapped[str | None] = mapped_column(String(200), nullable=True)  # -> configs.id (kind=data)
    params: Mapped[dict] = mapped_column(JSON, default=dict)   # max_nodes, filter_top25_dangerous, ...
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "source": self.source,
                "mode": self.mode, "num_classes": self.num_classes,
                "data_config_id": self.data_config_id, **(self.params or {})}


class JobRecord(Base):
    __tablename__ = "relearn_jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    method: Mapped[str] = mapped_column(String(32))
    dataset_ids: Mapped[list] = mapped_column(JSON, default=list)
    base_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    config_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "status": self.status, "method": self.method,
                "dataset_ids": self.dataset_ids or [], "base_model_id": self.base_model_id,
                "config_path": self.config_path, "log_path": self.log_path,
                "result_model_id": self.result_model_id, "message": self.message}


class DatasetJobRecord(Base):
    """Async dataset-ingestion job (raw data -> CPG -> .pt -> object storage -> registered
    dataset). Run by a Celery worker; this row is the pollable status store."""
    __tablename__ = "dataset_jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    name: Mapped[str] = mapped_column(String(256), default="")
    dataset_id: Mapped[str | None] = mapped_column(String(128), nullable=True)  # set when done
    data_config: Mapped[dict] = mapped_column(JSON, default=dict)               # frozen build params
    num_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "status": self.status, "name": self.name,
                "dataset_id": self.dataset_id, "data_config": self.data_config or {},
                "num_rows": self.num_rows, "message": self.message,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None}


class GraphCache(Base):
    """Pointer to a cached Joern CPG. The blob itself lives in object storage (graphs
    bucket); the DB keeps only the hash -> object-key mapping (best practice: no blobs in
    SQL). Skips the slow Joern pass on repeat inputs."""
    __tablename__ = "graph_cache"
    code_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    fmt: Mapped[str] = mapped_column(String(16), default="graphml")
    object_key: Mapped[str] = mapped_column(Text)        # key in the graphs bucket
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class InferenceResult(Base):
    """Persisted prediction history. code_hash links to graph_cache (the built CPG)."""
    __tablename__ = "inference_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    code_hash: Mapped[str] = mapped_column(String(64), index=True)   # FK-ish -> graph_cache.code_hash
    prediction: Mapped[str | None] = mapped_column(String(32), nullable=True)
    class_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_vulnerable: Mapped[bool | None] = mapped_column(default=None, nullable=True)
    confidence: Mapped[float | None] = mapped_column(default=None, nullable=True)
    suspicious_lines: Mapped[list] = mapped_column(JSON, default=list)
    # Pre-head function representation (vector fed to the classification head),
    # for drift detection / similarity search. JSON keeps it portable (SQLite dev /
    # Postgres prod); a per-model pgvector column + ANN index is added at search time.
    cls_embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    cls_embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self, with_embedding: bool = False) -> dict:
        d = {"id": self.id, "model_id": self.model_id, "code_hash": self.code_hash,
             "prediction": self.prediction, "class_id": self.class_id,
             "is_vulnerable": self.is_vulnerable, "confidence": self.confidence,
             "suspicious_lines": self.suspicious_lines or [],
             "cls_embedding_dim": self.cls_embedding_dim,
             "created_at": self.created_at.isoformat() if self.created_at else None}
        if with_embedding:
            d["cls_embedding"] = self.cls_embedding
        return d
