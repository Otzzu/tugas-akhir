"""Centralised settings, all overridable via environment variables."""
from __future__ import annotations

import os
from pathlib import Path


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Settings:
    # paths
    ROOT: Path = Path(__file__).resolve().parents[2]        # repo root
    API_DIR: Path = Path(__file__).resolve().parents[1]     # API/
    SEEDS_DIR: Path = API_DIR / "seeds"
    JOBS_DIR: Path = API_DIR / "jobs"

    # database — sqlite for dev, postgres for prod
    DATABASE_URL: str = os.environ.get("DATABASE_URL", f"sqlite:///{API_DIR / 'app.db'}")

    # inference
    JOERN_CLI: str = os.environ.get("JOERN_CLI", "C:/joern/joern-cli")
    DEVICE: str = os.environ.get("API_DEVICE") or _default_device()
    MAX_NODES: int = int(os.environ.get("API_MAX_NODES", "2500"))

    # dataset file upload (POST /datasets/upload). The inline `rows` body stays capped at
    # 5000 by the schema; a file may carry more, but never an unbounded amount.
    MAX_UPLOAD_BYTES: int = int(os.environ.get("API_MAX_UPLOAD_BYTES", str(256 * 1024 * 1024)))
    MAX_UPLOAD_ROWS: int = int(os.environ.get("API_MAX_UPLOAD_ROWS", "50000"))

    # server
    CORS_ORIGINS: list[str] = os.environ.get("API_CORS_ORIGINS", "*").split(",")

    # object storage — large blobs (CPG graphs) live here, NOT in the DB.
    # "fs" = local filesystem (dev, zero deps); "s3" = MinIO/AWS via boto3 (prod).
    STORAGE_BACKEND: str = os.environ.get("STORAGE_BACKEND", "fs")
    STORAGE_DIR: Path = Path(os.environ.get("STORAGE_DIR", str(API_DIR / "storage")))
    S3_ENDPOINT: str = os.environ.get("S3_ENDPOINT", "")        # e.g. http://minio:9000
    S3_ACCESS_KEY: str = os.environ.get("S3_ACCESS_KEY", "minioadmin")
    S3_SECRET_KEY: str = os.environ.get("S3_SECRET_KEY", "minioadmin")
    S3_REGION: str = os.environ.get("S3_REGION", "us-east-1")
    S3_BUCKET_GRAPHS: str = os.environ.get("S3_BUCKET_GRAPHS", "graphs")
    S3_BUCKET_DATASETS: str = os.environ.get("S3_BUCKET_DATASETS", "datasets")
    S3_BUCKET_CHECKPOINTS: str = os.environ.get("S3_BUCKET_CHECKPOINTS", "checkpoints")

    # async jobs (Celery) — broker + result backend (Redis by default).
    CELERY_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")


settings = Settings()
