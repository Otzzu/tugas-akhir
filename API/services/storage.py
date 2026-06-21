"""
Object storage for large binary artifacts (CPG graphs, optionally datasets/checkpoints).

Two interchangeable backends, chosen by settings.STORAGE_BACKEND:
  - "fs": local filesystem under settings.STORAGE_DIR (dev default, zero deps, no MinIO).
  - "s3": MinIO / AWS S3 via boto3 (prod). Buckets auto-created on first write.

Same call surface either way, so prod swaps the backend with no code change. Keeping big
blobs out of Postgres is the documented best practice — the DB holds metadata + a pointer
(bucket/key) only, never the blob itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from API.core.config import settings


class _Backend(Protocol):
    def put(self, bucket: str, key: str, data: bytes) -> str: ...
    def get(self, bucket: str, key: str) -> bytes | None: ...
    def exists(self, bucket: str, key: str) -> bool: ...
    def uri(self, bucket: str, key: str) -> str: ...


class FilesystemBackend:
    """Dev fallback: <STORAGE_DIR>/<bucket>/<key>. No MinIO needed for local runs."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, bucket: str, key: str) -> Path:
        return self.root / bucket / key

    def put(self, bucket: str, key: str, data: bytes) -> str:
        p = self._path(bucket, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return self.uri(bucket, key)

    def get(self, bucket: str, key: str) -> bytes | None:
        p = self._path(bucket, key)
        return p.read_bytes() if p.exists() else None

    def exists(self, bucket: str, key: str) -> bool:
        return self._path(bucket, key).exists()

    def uri(self, bucket: str, key: str) -> str:
        return f"file://{self._path(bucket, key)}"


class S3Backend:
    """MinIO / AWS S3 via boto3. Endpoint + keys from settings; buckets auto-created."""

    def __init__(self):
        import boto3
        self._s3 = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT or None,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
        )
        self._ensured: set[str] = set()

    def _ensure(self, bucket: str) -> None:
        if bucket in self._ensured:
            return
        try:
            self._s3.head_bucket(Bucket=bucket)
        except Exception:
            try:
                self._s3.create_bucket(Bucket=bucket)
            except Exception:
                pass
        self._ensured.add(bucket)

    def put(self, bucket: str, key: str, data: bytes) -> str:
        self._ensure(bucket)
        self._s3.put_object(Bucket=bucket, Key=key, Body=data)
        return self.uri(bucket, key)

    def get(self, bucket: str, key: str) -> bytes | None:
        try:
            return self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception:
            return None

    def exists(self, bucket: str, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def uri(self, bucket: str, key: str) -> str:
        return f"s3://{bucket}/{key}"


_backend: _Backend | None = None


def backend() -> _Backend:
    global _backend
    if _backend is None:
        _backend = S3Backend() if settings.STORAGE_BACKEND == "s3" \
            else FilesystemBackend(settings.STORAGE_DIR)
    return _backend


# convenience pass-throughs
def put_bytes(bucket: str, key: str, data: bytes) -> str:
    return backend().put(bucket, key, data)


def get_bytes(bucket: str, key: str) -> bytes | None:
    return backend().get(bucket, key)


def exists(bucket: str, key: str) -> bool:
    return backend().exists(bucket, key)


def uri(bucket: str, key: str) -> str:
    return backend().uri(bucket, key)
