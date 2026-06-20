"""CPG graph cache (DB-backed). Keyed by sha256(code) — skips Joern on repeat inputs."""
from __future__ import annotations

import hashlib

from API.core.database import SessionLocal
from API.models.tables import GraphCache


def code_hash(code: str, lang: str = "") -> str:
    return hashlib.sha256((lang + "\n" + code).encode("utf-8")).hexdigest()


def get(h: str) -> tuple[str, str] | None:
    with SessionLocal() as s:
        g = s.get(GraphCache, h)
        return (g.fmt, g.content) if g else None


def put(h: str, fmt: str, content: str) -> None:
    with SessionLocal() as s:
        if s.get(GraphCache, h) is None:
            s.add(GraphCache(code_hash=h, fmt=fmt, content=content))
            s.commit()
