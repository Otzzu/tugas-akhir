"""
CPG graph cache. Keyed by sha256(code) — skips Joern on repeat inputs.

The CPG blob is stored in OBJECT STORAGE (graphs bucket); the DB keeps only the
hash -> object-key pointer. This keeps Postgres small and follows the documented
best practice of not putting large binaries in a relational DB.
"""
from __future__ import annotations

import hashlib

from API.core.config import settings
from API.core.database import SessionLocal
from API.models.tables import GraphCache
from API.services import storage

_BUCKET = settings.S3_BUCKET_GRAPHS


def code_hash(code: str, lang: str = "") -> str:
    return hashlib.sha256((lang + "\n" + code).encode("utf-8")).hexdigest()


def get(h: str) -> tuple[str, str] | None:
    """Return (fmt, content) or None. A pointer with a missing blob counts as a miss."""
    with SessionLocal() as s:
        g = s.get(GraphCache, h)
        if g is None:
            return None
        key, fmt = g.object_key, g.fmt
    data = storage.get_bytes(_BUCKET, key)
    if data is None:
        return None
    return fmt, data.decode("utf-8", errors="replace")


def put(h: str, fmt: str, content: str) -> None:
    key = f"{h}.{fmt}"
    data = content.encode("utf-8")
    storage.put_bytes(_BUCKET, key, data)
    with SessionLocal() as s:
        if s.get(GraphCache, h) is None:
            s.add(GraphCache(code_hash=h, fmt=fmt, object_key=key, size_bytes=len(data)))
            s.commit()
