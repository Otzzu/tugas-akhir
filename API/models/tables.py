"""SQLAlchemy ORM tables: models, datasets, configs, relearn_jobs, graph_cache."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import String, Integer, Text, JSON, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from API.core.database import Base


class ConfigRecord(Base):
    """Content-addressed, IMMUTABLE config. id = '{name}@{hash}'. One config per entity covers
    the whole flow (data + model + train); there is no kind taxonomy. Editing never rewrites a
    row — same content reuses the id (dedup), different content gets a new id, mirroring how
    datasets/models are append-only. `content` is the canonical YAML snapshot."""
    __tablename__ = "configs"
    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    arch: Mapped[str] = mapped_column(String(64), default="")
    path: Mapped[str] = mapped_column(Text, default="")             # original source path (informational)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arch": self.arch,
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
    dataset_ids: Mapped[list] = mapped_column(JSON, default=list)  # CUMULATIVE training-data lineage: ancestors' datasets + this job's (chronological, deduped) — ER replay pool reads this
    val_dataset_id: Mapped[str | None] = mapped_column(String(128), nullable=True)   # role dataset (val)
    test_dataset_id: Mapped[str | None] = mapped_column(String(128), nullable=True)  # role dataset (test/benchmark)
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
            "dataset_ids": self.dataset_ids or [],
            "val_dataset_id": self.val_dataset_id, "test_dataset_id": self.test_dataset_id,
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


class RawDatasetRecord(Base):
    """The exact raw rows a dataset was built from — first-class entity so one raw can
    back many datasets (same rows, different build configs). Content-addressed id gives
    dedup for free: re-uploading identical rows reuses the same raw row + S3 object.
    The DB holds only metadata + the pointer; the JSON blob lives in object storage."""
    __tablename__ = "raw_datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)      # "raw_" + sha256(bytes)[:12]
    storage_uri: Mapped[str] = mapped_column(Text)                     # -> datasets bucket, {id}.json
    num_rows: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), default="")  # full sha256
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {"id": self.id, "storage_uri": self.storage_uri, "num_rows": self.num_rows,
                "size_bytes": self.size_bytes, "content_hash": self.content_hash}


class DatasetRecord(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(256), default="")
    source: Mapped[str] = mapped_column(String(128))
    mode: Mapped[str] = mapped_column(String(32), default="multiclass")
    num_classes: Mapped[int] = mapped_column(Integer, default=2)
    data_config_id: Mapped[str | None] = mapped_column(String(200), nullable=True)  # -> configs.id
    raw_id: Mapped[str | None] = mapped_column(ForeignKey("raw_datasets.id"), nullable=True)  # the raw rows this .pt was built from
    source_dataset_ids: Mapped[list] = mapped_column(JSON, default=list)  # merge provenance -> datasets.id, each with its own raw_id
    params: Mapped[dict] = mapped_column(JSON, default=dict)   # max_nodes, filter_top25_dangerous, ...
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "source": self.source,
                "mode": self.mode, "num_classes": self.num_classes,
                "data_config_id": self.data_config_id,
                "raw_id": self.raw_id, "source_dataset_ids": self.source_dataset_ids or [],
                **(self.params or {})}


class JobRecord(Base):
    __tablename__ = "relearn_jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    method: Mapped[str] = mapped_column(String(32))
    dataset_ids: Mapped[list] = mapped_column(JSON, default=list)
    base_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    val_dataset_id: Mapped[str | None] = mapped_column(String(128), nullable=True)   # role dataset (val)
    test_dataset_id: Mapped[str | None] = mapped_column(String(128), nullable=True)  # role dataset (test/benchmark)
    config_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)   # compact eval summary (task-B test)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "status": self.status, "method": self.method,
                "dataset_ids": self.dataset_ids or [], "base_model_id": self.base_model_id,
                "val_dataset_id": self.val_dataset_id, "test_dataset_id": self.test_dataset_id,
                "config_path": self.config_path, "log_path": self.log_path,
                "result_model_id": self.result_model_id, "message": self.message,
                "metrics": self.metrics}


class DatasetJobRecord(Base):
    """Async dataset-ingestion job (raw data -> CPG -> .pt -> object storage -> registered
    dataset). Run by a Celery worker; this row is the pollable status store."""
    __tablename__ = "dataset_jobs"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    name: Mapped[str] = mapped_column(String(256), default="")
    dataset_id: Mapped[str | None] = mapped_column(String(128), nullable=True)  # set when done
    raw_id: Mapped[str | None] = mapped_column(String(64), nullable=True)       # -> raw_datasets, set when done
    data_config: Mapped[dict] = mapped_column(JSON, default=dict)               # frozen build params
    num_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "status": self.status, "name": self.name,
                "dataset_id": self.dataset_id, "raw_id": self.raw_id,
                "data_config": self.data_config or {},
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
