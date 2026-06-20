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

    # server
    CORS_ORIGINS: list[str] = os.environ.get("API_CORS_ORIGINS", "*").split(",")


settings = Settings()
