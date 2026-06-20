"""SQLAlchemy ORM tables: models, datasets, relearn_jobs, graph_cache."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import String, Integer, Text, JSON, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from API.core.database import Base


class ModelRecord(Base):
    __tablename__ = "models"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(256), default="")
    arch: Mapped[str] = mapped_column(String(64))
    config: Mapped[str] = mapped_column(Text)
    checkpoint: Mapped[str] = mapped_column(Text)
    dataset_id: Mapped[str] = mapped_column(String(128), default="")
    num_classes: Mapped[int] = mapped_column(Integer, default=2)
    class_names: Mapped[list] = mapped_column(JSON, default=list)
    base_model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label, "arch": self.arch, "config": self.config,
            "checkpoint": self.checkpoint, "dataset_id": self.dataset_id,
            "num_classes": self.num_classes, "class_names": self.class_names or [],
            "base_model_id": self.base_model_id, "method": self.method,
        }


class DatasetRecord(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(256), default="")
    source: Mapped[str] = mapped_column(String(128))
    mode: Mapped[str] = mapped_column(String(32), default="multiclass")
    num_classes: Mapped[int] = mapped_column(Integer, default=2)
    params: Mapped[dict] = mapped_column(JSON, default=dict)   # max_nodes, filter_top25_dangerous, ...
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "source": self.source,
                "mode": self.mode, "num_classes": self.num_classes, **(self.params or {})}


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


class GraphCache(Base):
    """Cached Joern CPG output keyed by code hash — skips the slow Joern pass on repeats."""
    __tablename__ = "graph_cache"
    code_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    fmt: Mapped[str] = mapped_column(String(16), default="graphml")
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
